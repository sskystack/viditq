#!/bin/bash

fp_path="/root/autodl-tmp/viditq/examples/flux/logs/final/BASIC8/generated_images_30"
base_dir="/root/autodl-tmp/viditq/examples/flux/logs/final/Perrow_QKsmooth8_PV8/generated_images_30"
log_file="./logs/test.txt"

python fid_score.py \
        --path "$fp_path" "$base_dir" \
        --log_file "$log_file"
# Full Suite PTQ FID eval
# for dir in "$base_dir"/*; do
#     if [ -d "$dir" ]; then
#         echo "FID evaluation: QUANT: $dir FP: $fp_path"
#         python evaluation/fid_score.py \
#         --path "$fp_path" "$dir" \
#         --log_file "$log_file"
#     fi
# done

