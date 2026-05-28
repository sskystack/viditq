import argparse
import os

import torch
from mmengine.config import Config
from tqdm import tqdm

from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, build_module
from opensora.utils.inference_utils import (
    append_score_to_prompts,
    extract_json_from_prompts,
    extract_prompts_loop,
    load_prompts,
    merge_prompt,
    split_prompt,
)
from opensora.utils.misc import to_torch_dtype


def parse_args():
    parser = argparse.ArgumentParser(description="Precompute Open-Sora T5 text embeddings for a prompt file.")
    parser.add_argument("config", help="Open-Sora config file")
    parser.add_argument("--prompt-path", required=True, help="Prompt txt file")
    parser.add_argument("--output", required=True, help="Output .pth file")
    parser.add_argument("--batch-size", type=int, default=None, help="Text encoder batch size")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--device", default="cuda", help="Device for the T5 encoder")
    parser.add_argument(
        "--device-map",
        default=None,
        choices=[None, "auto", "balanced", "balanced_low_0", "sequential"],
        help="Transformers device_map for sharding T5 across visible GPUs",
    )
    parser.add_argument(
        "--max-memory-per-gpu",
        default=None,
        help='Per-GPU max memory for device_map, e.g. "14GiB"',
    )
    parser.add_argument(
        "--offload-folder",
        default=None,
        help="Optional CPU/disk offload folder used by Transformers device_map",
    )
    return parser.parse_args()


def build_processed_prompts(cfg, prompt_path, start_index, end_index):
    raw_prompts = load_prompts(prompt_path, start_index, end_index)
    reference_path = [""] * len(raw_prompts)
    mask_strategy = [""] * len(raw_prompts)
    prompts, _, _ = extract_json_from_prompts(raw_prompts, reference_path, mask_strategy)

    processed = []
    loop = cfg.get("loop", 1)
    for prompt in prompts:
        prompt_segment_list, loop_idx_list = split_prompt(prompt)
        prompt_segment_list = append_score_to_prompts(
            prompt_segment_list,
            aes=cfg.get("aes", None),
            flow=cfg.get("flow", None),
            camera_motion=cfg.get("camera_motion", None),
        )
        prompt_segment_list = [text_preprocessing(text) for text in prompt_segment_list]
        merged_prompt = merge_prompt(prompt_segment_list, loop_idx_list)
        for loop_i in range(loop):
            processed.append(extract_prompts_loop([merged_prompt], loop_i)[0])

    return list(dict.fromkeys(processed))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    batch_size = args.batch_size or cfg.get("text_embed_batch_size", cfg.get("batch_size", 1))

    cfg.text_encoder["dtype"] = dtype
    if args.device_map is not None:
        t5_model_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
            "device_map": args.device_map,
        }
        if args.max_memory_per_gpu is not None:
            gpu_count = torch.cuda.device_count()
            t5_model_kwargs["max_memory"] = {i: args.max_memory_per_gpu for i in range(gpu_count)}
        if args.offload_folder is not None:
            os.makedirs(args.offload_folder, exist_ok=True)
            t5_model_kwargs["offload_folder"] = args.offload_folder
        cfg.text_encoder["t5_model_kwargs"] = t5_model_kwargs

    text_encoder = build_module(cfg.text_encoder, MODELS, device=args.device)
    if args.device_map is None:
        text_encoder.t5.model.to(dtype=dtype)

    prompts = build_processed_prompts(cfg, args.prompt_path, args.start_index, args.end_index)
    text_embeds = {}
    for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding prompts"):
        batch_prompts = prompts[i : i + batch_size]
        model_args = text_encoder.encode(batch_prompts)
        y = model_args["y"].detach().cpu()
        mask = model_args["mask"].detach().cpu()
        for j, prompt in enumerate(batch_prompts):
            text_embeds[prompt] = {
                "y": y[j : j + 1].contiguous(),
                "mask": mask[j : j + 1].contiguous(),
            }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    torch.save(
        {
            "prompt_path": args.prompt_path,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "prompts": prompts,
            "text_embeds": text_embeds,
        },
        args.output,
    )
    print(f"Saved {len(text_embeds)} text embeddings to {args.output}")


if __name__ == "__main__":
    main()
