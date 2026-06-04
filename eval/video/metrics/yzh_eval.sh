dir_videos=/root/autodl-tmp/viditq/examples/opensora1.2/logs/motion_ptq_vbench


CUDA_VISIBLE_DEVICES=1 python3 clip_score.py --dir_videos $dir_videos --dir_prompts $dir_prompts --dir_results $dir_results --metric 'clip_temp_score'
