import math

import torch
import torch.nn.functional as F

from qdiff.opensora_motion.motion_context import motion_context
from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear


def _dynamic_symmetric_quantize(x, n_bits, clip_value=None, eps=1.0e-6):
    if x.numel() == 0:
        return x
    n_levels = 2 ** (int(n_bits) - 1) - 1
    x_absmax = x.abs().amax(dim=1, keepdim=True).clamp(min=eps)
    if clip_value is not None:
        clip_value = torch.as_tensor(clip_value, device=x.device, dtype=x.dtype).reshape(1, 1).clamp(min=eps)
        x_absmax = torch.minimum(x_absmax, clip_value)
    delta = x_absmax / n_levels
    x_int = torch.round(x / delta).clamp(-n_levels - 1, n_levels)
    return x_int * delta


class MotionQuantizedLinear(ViDiTQuantizedLinear):
    """OpenSora-only ViDiT-Q layer with motion-routed activation bit-widths."""

    def __init__(self, in_features, out_features, bias, device, quant_config, fp_module):
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)
        motion_cfg = quant_config.get("motion_ptq", None)
        self.low_bit = int(motion_cfg.get("low_bit", 4)) if motion_cfg is not None else 4
        self.mid_bit = int(motion_cfg.get("mid_bit", 6)) if motion_cfg is not None else 6
        self.high_bit = int(motion_cfg.get("high_bit", 8)) if motion_cfg is not None else 8
        self.base8_high_ratio = float(motion_cfg.get("base8_high_ratio", 0.85)) if motion_cfg is not None else 0.85
        self.base8_mid_ratio = float(motion_cfg.get("base8_mid_ratio", 0.12)) if motion_cfg is not None else 0.12
        self.base4_high_ratio = float(motion_cfg.get("base4_high_ratio", 0.15)) if motion_cfg is not None else 0.15
        self.base4_mid_ratio = float(motion_cfg.get("base4_mid_ratio", 0.25)) if motion_cfg is not None else 0.25
        self.motion_weight = float(motion_cfg.get("motion_weight", 0.7)) if motion_cfg is not None else 0.7
        self.activation_weight = float(motion_cfg.get("activation_weight", 0.3)) if motion_cfg is not None else 0.3
        self.motion_calibration = False
        self.clip_quantile = float(motion_cfg.get("clip_quantile", 0.999)) if motion_cfg is not None else 0.999
        self.motion_clip_stats = {}
        self.motion_clip = {}

    def set_motion_calibration(self, enabled=True, reset=False, clip_quantile=None):
        self.motion_calibration = enabled
        if clip_quantile is not None:
            self.clip_quantile = float(clip_quantile)
        if reset:
            self.motion_clip_stats = {}

    def get_motion_clip(self):
        if self.motion_clip_stats:
            return {str(bit): value.detach().cpu() for bit, value in self.motion_clip_stats.items()}
        return {str(bit): torch.as_tensor(value).detach().cpu() for bit, value in self.motion_clip.items()}

    def load_motion_clip(self, clip_dict):
        self.motion_clip = {int(bit): torch.as_tensor(value) for bit, value in clip_dict.items()}

    def _observe_motion_clip(self, x, bits):
        if not self.motion_calibration:
            return
        channel = x.shape[-1]
        x_flat = x.detach().reshape(-1, channel).float()
        bits_flat = bits.reshape(-1).to(x_flat.device)
        for n_bits in (self.low_bit, self.mid_bit, self.high_bit):
            mask = bits_flat == n_bits
            if not torch.any(mask):
                continue
            token_absmax = x_flat[mask].abs().amax(dim=1)
            clip_value = torch.quantile(token_absmax, self.clip_quantile)
            old_value = self.motion_clip_stats.get(n_bits, None)
            if old_value is None:
                self.motion_clip_stats[n_bits] = clip_value.detach()
            else:
                self.motion_clip_stats[n_bits] = torch.maximum(old_value.to(clip_value.device), clip_value.detach())

    def _current_base_act_bit(self):
        if self.a_quantizer is None:
            return self.high_bit
        return int(getattr(self.a_quantizer, "n_bits", self.high_bit))

    def _ratio_for_base_bit(self, base_bit):
        if base_bit <= self.low_bit:
            return self.base4_high_ratio, self.base4_mid_ratio
        return self.base8_high_ratio, self.base8_mid_ratio

    @staticmethod
    def _normalize_token_score(score):
        flat = score.reshape(score.shape[0], -1)
        median = flat.median(dim=1).values.reshape(score.shape[0], 1).clamp(min=1.0e-6)
        return score / median

    def _route_bits(self, score, base_bit):
        high_ratio, mid_ratio = self._ratio_for_base_bit(base_bit)
        high_ratio = max(0.0, min(1.0, high_ratio))
        mid_ratio = max(0.0, min(1.0 - high_ratio, mid_ratio))

        flat_score = score.reshape(score.shape[0], -1)
        num_tokens = flat_score.shape[1]
        num_high = int(math.ceil(num_tokens * high_ratio))
        num_mid = int(math.ceil(num_tokens * mid_ratio))
        num_high = max(0, min(num_tokens, num_high))
        num_high_mid = max(0, min(num_tokens, num_high + num_mid))

        flat_bits = torch.full_like(flat_score, self.low_bit, dtype=torch.int16)
        if num_high > 0:
            high_threshold = torch.topk(flat_score, k=num_high, dim=1).values[:, -1].reshape(score.shape[0], 1)
            flat_bits = torch.where(
                flat_score >= high_threshold,
                torch.full_like(flat_bits, self.high_bit),
                flat_bits,
            )
        if num_high_mid > num_high:
            mid_threshold = torch.topk(flat_score, k=num_high_mid, dim=1).values[:, -1].reshape(score.shape[0], 1)
            mid_mask = (flat_score >= mid_threshold) & (flat_bits != self.high_bit)
            flat_bits = torch.where(
                mid_mask,
                torch.full_like(flat_bits, self.mid_bit),
                flat_bits,
            )
        return flat_bits.reshape(score.shape)

    def _motion_quantize_activation(self, x):
        motion_score = motion_context.resolve_scores(x.shape)
        if motion_score is None or self.a_quantizer is None:
            bsz, token_num, channel = x.shape
            x_flat = x.reshape(bsz * token_num, channel)
            return self.a_quantizer(x_flat).reshape_as(x) if self.a_quantizer is not None else x

        base_bit = self._current_base_act_bit()
        token_absmax = x.detach().abs().amax(dim=-1).float()
        motion_score = self._normalize_token_score(motion_score.to(device=x.device, dtype=torch.float32))
        activation_score = self._normalize_token_score(token_absmax)
        score = self.motion_weight * motion_score + self.activation_weight * activation_score
        bits = self._route_bits(score, base_bit)
        self._observe_motion_clip(x, bits)
        dtype = x.dtype
        channel = x.shape[-1]
        x_flat = x.reshape(-1, channel)
        bits_flat = bits.reshape(-1).to(x_flat.device)
        out = torch.empty_like(x_flat)
        for n_bits in (self.low_bit, self.mid_bit, self.high_bit):
            mask = bits_flat == n_bits
            if torch.any(mask):
                clip_value = self.motion_clip.get(n_bits, None)
                out[mask] = _dynamic_symmetric_quantize(x_flat[mask].float(), n_bits, clip_value=clip_value).to(dtype)

        other = (bits_flat != self.low_bit) & (bits_flat != self.mid_bit) & (bits_flat != self.high_bit)
        if torch.any(other):
            clip_value = self.motion_clip.get(self.low_bit, None)
            out[other] = _dynamic_symmetric_quantize(x_flat[other].float(), self.low_bit, clip_value=clip_value).to(dtype)
        return out.reshape_as(x)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if not self.quant_mode:
            return self.fp_module(x, *args, **kwargs)

        dtype_ = x.dtype
        if self.channel_mask is not None and self.rotation_matrix is not None:
            channel = x.shape[-1]
            x = x * self.channel_mask.reshape([1, 1, channel]).to(device=x.device, dtype=x.dtype)
            x = torch.matmul(x.double(), self.rotation_matrix).to(dtype=dtype_)

        x = self._motion_quantize_activation(x)
        return F.linear(x, self.weight.to(dtype=dtype_), self.bias, *args, **kwargs)
