import math

import torch
import torch.nn.functional as F


def _cfg_get(cfg, name, default):
    if cfg is None:
        return default
    return cfg.get(name, default) if hasattr(cfg, "get") else getattr(cfg, name, default)


def _as_3tuple(value):
    if isinstance(value, int):
        return (value, value, value)
    return tuple(value)


def _timestep_gate(timestep, cfg):
    gate = _cfg_get(cfg, "timestep_gate", "none")
    if gate in (None, "none"):
        return None
    if gate == "middle_late":
        threshold = float(_cfg_get(cfg, "timestep_threshold", 0.25))
        max_timestep = float(_cfg_get(cfg, "max_timestep", 1000.0))
        t = timestep.float() / max_timestep
        return (t <= (1.0 - threshold)).float()
    return None


@torch.no_grad()
def compute_motion_token_scores(latent, timestep, num_frames, height, width, patch_size, cfg):
    """Return normalized motion scores with shape [B, T, S] for OpenSora STDiT3 tokens."""
    macroblock_size = int(_cfg_get(cfg, "macroblock_size", 4))
    first_policy = _cfg_get(cfg, "first_frame_policy", "repeat_next")
    score_reduce = _cfg_get(cfg, "score_reduce", "max")

    bsz, _, raw_t, _, _ = latent.shape
    if raw_t <= 1:
        return torch.zeros((bsz, num_frames, height * width), device=latent.device, dtype=torch.float32)

    diff = (latent[:, :, 1:] - latent[:, :, :-1]).abs().mean(dim=1)
    if first_policy == "zero":
        first = torch.zeros_like(diff[:, :1])
    else:
        first = diff[:, :1]
    diff = torch.cat([first, diff], dim=1).unsqueeze(1)

    patch_t, patch_h, patch_w = _as_3tuple(patch_size)
    score = F.avg_pool3d(diff.float(), kernel_size=(patch_t, patch_h, patch_w), stride=(patch_t, patch_h, patch_w), ceil_mode=True)
    score = score[:, 0]
    if score.shape[-3:] != (num_frames, height, width):
        score = F.interpolate(score.unsqueeze(1), size=(num_frames, height, width), mode="trilinear", align_corners=False)[:, 0]

    gate = _timestep_gate(timestep, cfg)
    if gate is not None:
        score = score * gate.reshape(bsz, 1, 1, 1)

    flat_score = score.reshape(bsz, -1)
    norm = flat_score.median(dim=1).values.reshape(bsz, 1, 1, 1).clamp(min=1.0e-6)
    score = score / norm

    pad_h = (macroblock_size - height % macroblock_size) % macroblock_size
    pad_w = (macroblock_size - width % macroblock_size) % macroblock_size
    score_2d = score.reshape(bsz * num_frames, 1, height, width)
    if pad_h or pad_w:
        score_2d = F.pad(score_2d, (0, pad_w, 0, pad_h), mode="replicate")
    pool_fn = F.max_pool2d if score_reduce == "max" else F.avg_pool2d
    block_score = pool_fn(score_2d, kernel_size=macroblock_size, stride=macroblock_size)

    block_h = (height + pad_h) // macroblock_size
    block_w = (width + pad_w) // macroblock_size
    score_2d = block_score.reshape(bsz * num_frames, 1, block_h, block_w).float()
    score_2d = F.interpolate(score_2d, size=(height, width), mode="nearest")
    return score_2d.reshape(bsz, num_frames, height * width)


@torch.no_grad()
def compute_motion_token_bits(latent, timestep, num_frames, height, width, patch_size, cfg):
    """Return token bit-widths with shape [B, T, S] for compatibility."""
    low_bit = int(_cfg_get(cfg, "low_bit", 4))
    mid_bit = int(_cfg_get(cfg, "mid_bit", 6))
    high_bit = int(_cfg_get(cfg, "high_bit", 8))
    high_ratio = float(_cfg_get(cfg, "high_ratio", 0.2))
    mid_ratio = float(_cfg_get(cfg, "mid_ratio", 0.3))
    scores = compute_motion_token_scores(latent, timestep, num_frames, height, width, patch_size, cfg)
    flat_score = scores.reshape(scores.shape[0], -1)
    num_tokens = flat_score.shape[1]
    num_high = max(1, int(math.ceil(num_tokens * high_ratio)))
    num_mid = max(0, int(math.ceil(num_tokens * mid_ratio)))
    num_high_mid = min(num_tokens, num_high + num_mid)
    high_threshold = torch.topk(flat_score, k=num_high, dim=1).values[:, -1].reshape(scores.shape[0], 1)
    if num_high_mid > num_high:
        mid_threshold = torch.topk(flat_score, k=num_high_mid, dim=1).values[:, -1].reshape(scores.shape[0], 1)
    else:
        mid_threshold = high_threshold
    flat_bits = torch.where(
        flat_score >= high_threshold,
        torch.full_like(flat_score, high_bit, dtype=torch.int16),
        torch.where(
            flat_score >= mid_threshold,
            torch.full_like(flat_score, mid_bit, dtype=torch.int16),
            torch.full_like(flat_score, low_bit, dtype=torch.int16),
        ),
    )
    return flat_bits.reshape_as(scores).to(torch.int16)
