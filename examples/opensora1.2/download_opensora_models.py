import argparse
import os
from pathlib import Path


REQUIRED_REPOS = [
    ("hpcai-tech/OpenSora-STDiT-v3", "hpcai-tech/OpenSora-STDiT-v3", None),
    ("hpcai-tech/OpenSora-VAE-v1.2", "hpcai-tech/OpenSora-VAE-v1.2", None),
    ("PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", ["vae/*"]),
]

T5_REPO = ("DeepFloyd/t5-v1_1-xxl", "DeepFloyd/t5-v1_1-xxl", None)


def parse_args():
    parser = argparse.ArgumentParser(description="Download Open-Sora v1.2 models with huggingface_hub.")
    parser.add_argument(
        "--output-dir",
        default="./models",
        help="Directory to save models. Upload this directory to /root/data-fs/models on the server.",
    )
    parser.add_argument(
        "--endpoint",
        default="https://hf-mirror.com",
        help="Hugging Face endpoint or mirror. Use https://huggingface.co for the official endpoint.",
    )
    parser.add_argument(
        "--include-t5",
        action="store_true",
        help="Also download DeepFloyd/t5-v1_1-xxl for local/server text-embedding precompute.",
    )
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--enable-xet", action="store_true", help="Enable hf-xet downloads. Disabled by default for mirrors.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["HF_ENDPOINT"] = args.endpoint
    if not args.enable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"

    from huggingface_hub import snapshot_download

    repos = list(REQUIRED_REPOS)
    if args.include_t5:
        repos.append(T5_REPO)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for repo_id, relative_path, allow_patterns in repos:
        local_dir = output_dir / relative_path
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nDownloading {repo_id}")
        print(f"  -> {local_dir}")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            token=args.token,
            max_workers=args.max_workers,
            allow_patterns=allow_patterns,
        )

    print("\nDone. Upload the output directory to the server with:")
    print(f"rsync -avP -e \"ssh -p 42102\" {output_dir}/ root@10.137.144.40:/root/data-fs/models/")


if __name__ == "__main__":
    main()
