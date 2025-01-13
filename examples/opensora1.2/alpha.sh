GPU_ID="5"

CUDA_VISIBLE_DEVICES=$GPU_ID python alpha.py configs/sample.py \
  --num-frames 4s --resolution 240p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt