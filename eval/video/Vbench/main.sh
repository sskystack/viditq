export HF_ENDPOINT=https://hf-mirror.com
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/root/autodl-tmp/viditq/eval/video/Vbench/.cache/vm_ckpt}"

current_time=$(date +"%Y%m%d%H%M%S")
JSON="/root/autodl-tmp/viditq/eval/video/Vbench/vbench/VBench_full_info.json"
VIDEO_PATH="/root/autodl-tmp/viditq/examples/opensora1.2/logs/motion_ptq_vbench"
OUTPUT="./evaluation_results/motion_ptq_vbench_${current_time}"

# If torch.hub.load reports an error, try using local download by setting --load_ckpt_from_local to True
python evaluate.py --output_path "$OUTPUT" --full_json_dir "$JSON" --dimension "subject_consistency" --videos_path "$VIDEO_PATH" --load_ckpt_from_local True
python evaluate.py --output_path "$OUTPUT" --full_json_dir "$JSON" --dimension "overall_consistency" --videos_path "$VIDEO_PATH"
python evaluate.py --output_path "$OUTPUT" --full_json_dir "$JSON" --dimension "scene" --videos_path "$VIDEO_PATH"
