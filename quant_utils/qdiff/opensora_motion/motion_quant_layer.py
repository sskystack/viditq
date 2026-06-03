import torch
import torch.nn.functional as F

from qdiff.base.quant_layer import QuantizedLinear
from qdiff.opensora_motion.motion_context import motion_context


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


class MotionQuantizedLinear(QuantizedLinear):
    """OpenSora-only motion-routed activation quantization layer."""

    def __init__(self, in_features, out_features, bias, device, quant_config, fp_module):
        super().__init__(in_features, out_features, bias, device, quant_config, fp_module)
        motion_cfg = quant_config.get("motion_ptq", None)
        self.low_bit = int(motion_cfg.get("low_bit", 4)) if motion_cfg is not None else 4
        self.mid_bit = int(motion_cfg.get("mid_bit", 6)) if motion_cfg is not None else 6
        self.high_bit = int(motion_cfg.get("high_bit", 8)) if motion_cfg is not None else 8
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

    def _motion_quantize_activation(self, x):
        bits = motion_context.resolve_bits(x.shape)
        if bits is None or self.a_quantizer is None:
            bsz, token_num, channel = x.shape
            x_flat = x.reshape(bsz * token_num, channel)
            return self.a_quantizer(x_flat).reshape_as(x) if self.a_quantizer is not None else x

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

        x = self._motion_quantize_activation(x)
        return F.linear(x, self.weight.to(dtype=x.dtype), self.bias, *args, **kwargs)
