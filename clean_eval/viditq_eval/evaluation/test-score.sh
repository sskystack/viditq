#!/bin/bash

# Define paths and arguments
export HF_ENDPOINT=https://hf-mirror.com
metric="CLIP"
dir="/share/public/zhuhongyu/diffuser-dev/examples/pixart/logs/coco_fp16/generated_images"
log_file="./CLIP_FP16.txt"
prompt_path="/share/public/zhuhongyu/diffuser-dev/examples/pixart/coco_1024.txt"
# for dir in "$base_dir"/*; do
#     if [ -d "$dir" ]; then
#         echo "$metric evaluation: QUANT: $dir"
#         python evaluation/test_score.py \
#         --prompts_path "$prompt_path" \
#         --metric "$metric" \
#         --img_dir "$dir" \
#         --log_file "$log_file"
#     fi
# done
python test_score.py \
    --prompts_path $prompt_path \
    --metric $metric \
    --img_dir $dir \
    --log_file $log_file