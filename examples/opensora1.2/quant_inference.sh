export HF_ENDPOINT=https://hf-mirror.com
GPU_ID="1"

CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py configs/sample.py \
  --num-frames 4s --resolution 144p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt

CUDA_VISIBLE_DEVICES=$GPU_ID python quant_inference.py configs/sample.py \
  --num-frames 4s --resolution 144p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt
