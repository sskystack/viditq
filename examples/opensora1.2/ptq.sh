export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES="1" python ptq.py configs/sample.py \
  --num-frames 4s --resolution 720p --aspect-ratio 9:16 \
  --prompt "a beautiful waterfall"