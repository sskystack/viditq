import logging
import os
import re

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.stdit.stdit3 import STDiT3
from opensora.utils.ckpt_utils import load_checkpoint
from qdiff.base.base_quantizer import BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.base.quant_model import (
    bitwidth_refactor_,
    load_quant_param_dict_,
    save_quant_param_dict_,
    set_init_done_,
)
from qdiff.opensora_motion.motion_context import motion_context
from qdiff.opensora_motion.motion_quant_layer import MotionQuantizedLinear
from qdiff.opensora_motion.motion_utils import compute_motion_token_scores
from qdiff.utils import apply_func_to_submodules

logger = logging.getLogger(__name__)


def motion_quant_layer_refactor_(submodule, name, parent_module, quant_config, full_name, remain_fp_regex):
    if remain_fp_regex is not None and re.search(re.compile(remain_fp_regex), full_name):
        logger.info(f"remain {full_name} quant as FP due to fp_regex")
        return

    quant_layer_type = QuantizedLinear
    motion_cfg = quant_config.get("motion_ptq", None)
    if motion_cfg is not None and motion_cfg.get("enabled", False):
        layer_regex = motion_cfg.get("layer_name_regex", "")
        if layer_regex and re.search(re.compile(layer_regex), full_name):
            quant_layer_type = MotionQuantizedLinear
            logger.info(f"setting motion PTQ for layer {full_name}")

    in_features = submodule.in_features
    out_features = submodule.out_features
    bias = submodule.bias is not None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(parent_module, name, quant_layer_type(in_features, out_features, bias, device, quant_config, submodule))

    quant_layer = getattr(parent_module, name)
    setattr(quant_layer, "module_name", full_name)
    if quant_layer.w_quantizer is not None:
        setattr(quant_layer.w_quantizer, "module_name", full_name)
    if quant_layer.a_quantizer is not None:
        setattr(quant_layer.a_quantizer, "module_name", full_name)


def set_motion_calibration_(submodule, enabled, reset, clip_quantile, observe_only):
    submodule.set_motion_calibration(
        enabled=enabled,
        reset=reset,
        clip_quantile=clip_quantile,
        observe_only=observe_only,
    )


def get_motion_clip_(submodule, full_name):
    return submodule.get_motion_clip()


def load_motion_clip_(submodule, full_name, clip_dict):
    if full_name in clip_dict:
        submodule.load_motion_clip(clip_dict[full_name])


class QuantOpenSoraMotion(STDiT3):
    def __init__(self, quant_config, config, from_pretrained):
        super().__init__(config)
        if from_pretrained is not None:
            hf_checkpoint = os.path.join(from_pretrained, "model.safetensors")
            if os.path.isdir(from_pretrained) and os.path.isfile(hf_checkpoint):
                from_pretrained = hf_checkpoint
            load_checkpoint(self, from_pretrained)

        self.quant_config = quant_config
        self.quant_param_dict = {}
        self.quant_layer_refactor()

    def quant_layer_refactor(self):
        apply_func_to_submodules(
            self,
            class_type=nn.Linear,
            function=motion_quant_layer_refactor_,
            name=None,
            parent_module=None,
            quant_config=self.quant_config,
            full_name=None,
            remain_fp_regex=self.quant_config.remain_fp_regex,
        )

    def save_quant_param_dict(self):
        apply_func_to_submodules(
            self,
            class_type=BaseQuantizer,
            function=save_quant_param_dict_,
            full_name=None,
            parent_module=None,
            model=self,
        )

    def load_quant_param_dict(self, quant_param_dict):
        apply_func_to_submodules(
            self,
            class_type=BaseQuantizer,
            function=load_quant_param_dict_,
            full_name=None,
            parent_module=None,
            quant_param_dict=quant_param_dict,
            model=self,
        )

    def set_init_done(self):
        apply_func_to_submodules(self, class_type=BaseQuantizer, function=set_init_done_)

    def bitwidth_refactor(self):
        apply_func_to_submodules(
            self,
            class_type=QuantizedLinear,
            function=bitwidth_refactor_,
            name=None,
            parent_module=None,
            quant_config=self.quant_config,
            full_name=None,
        )

    def enable_motion_calibration(self, enabled=True, reset=False, observe_only=False):
        motion_cfg = self.quant_config.get("motion_ptq", None)
        clip_quantile = motion_cfg.get("clip_quantile", 0.999) if motion_cfg is not None else 0.999
        apply_func_to_submodules(
            self,
            class_type=MotionQuantizedLinear,
            function=set_motion_calibration_,
            enabled=enabled,
            reset=reset,
            clip_quantile=clip_quantile,
            observe_only=observe_only,
        )

    def get_motion_clip_dict(self):
        clip_dict = apply_func_to_submodules(
            self,
            class_type=MotionQuantizedLinear,
            function=get_motion_clip_,
            return_d={},
            full_name=None,
        )
        return {name: clips for name, clips in clip_dict.items() if len(clips) > 0}

    def load_motion_clip_dict(self, clip_dict):
        apply_func_to_submodules(
            self,
            class_type=MotionQuantizedLinear,
            function=load_motion_clip_,
            full_name=None,
            clip_dict=clip_dict,
        )

    def _update_motion_context(self, x, timestep, num_frames, num_spatial_tokens, height, width):
        motion_cfg = self.quant_config.get("motion_ptq", None)
        if motion_cfg is None or not motion_cfg.get("enabled", False):
            motion_context.clear()
            return
        token_scores = compute_motion_token_scores(
            latent=x,
            timestep=timestep,
            num_frames=num_frames,
            height=height,
            width=width,
            patch_size=self.patch_size,
            cfg=motion_cfg,
        )
        motion_context.set_scores(token_scores, num_frames, num_spatial_tokens)

    def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)

        if self.enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            h_pad_size = sp_size - H % sp_size if H % sp_size != 0 else 0
            if h_pad_size > 0:
                H += h_pad_size
                x = F.pad(x, (0, 0, 0, h_pad_size * self.patch_size[1]))

        S = H * W
        self._update_motion_context(x, timestep, T, S, H, W)

        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        t = self.t_embedder(timestep, dtype=x.dtype)
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        x = self.x_embedder(x)
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="down")
            S = S // dist.get_world_size(get_sequence_parallel_group())

        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
            x = auto_grad_checkpoint(spatial_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)
            x = auto_grad_checkpoint(temporal_block, x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)

        if self.enable_sequence_parallelism:
            x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=2, grad_scale="up")
            S = S * dist.get_world_size(get_sequence_parallel_group())
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)
        motion_context.clear()
        return x.to(torch.float32)
