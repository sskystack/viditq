export HF_ENDPOINT=https://hf-mirror.com
GPU_ID="1"

CUDA_VISIBLE_DEVICES=$GPU_ID python get_calib_data.py configs/sample.py \
  --num-frames 4s --resolution 144p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt
