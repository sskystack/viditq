export HF_ENDPOINT=https://hf-mirror.com
GPU_ID="5"

CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py configs/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt