export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES="1" python fp_inference.py configs/sample.py \
  --num-frames 4s --resolution 144p --aspect-ratio 9:16 \
  --prompt-path ./prompts.txt
