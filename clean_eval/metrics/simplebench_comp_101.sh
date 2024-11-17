dir_videos="/home/zhuhongyu/diffuser-dev/examples/opensora1.2/samples/samples"
#dir_prompts="/share/public/video_quant/wanrui/datasets/ucf101/videos/prompt_ucf.txt"       #for latte only
dir_prompts="/home/zhuhongyu/diffuser-dev/examples/opensora1.2/ucf_long_100_copy.txt"   #for stdit
dir_results="/home/zhuhongyu/diffuser-dev/examples/opensora1.2"
dir_save=$(dirname $dir_videos)
dir_ref_vid_simple="/share/public/video_quant/wanrui/datasets/ucf101/videos/selected/test_zhuhongyu"                 #fvd_ucf101
dir_ref_vid_fp_latte="/mnt/public/video_quant/wanrui/alt/diffuser-dev/logs/latte/debug_timestep/generated_videos"    #fvd_fp16
dir_ref_vid_fp_stdit="/share/public/video_quant/wanrui/diffuser-dev/logs/opensora/test_zhuhongyu"

#EC_path="/share/public/video_quant/wanrui/metrics/I2VBench"
SIMP_path="/home/zhuhongyu/clean_eval/metrics"
CUDA_DEVICES="3"

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
dir_results="${dir_results}/${current_time}"
rm $dir_save/metrics.log
rm $dir_videos/metrics.log
mkdir -p "$dir_results"

# FVD
cd $SIMP_path

#fvd_ucf for latte
#CUDA_VISIBLE_DEVICES='2' python3 fvd.py --dir_videos $dir_videos --dir_results $dir_results --dir_ref_vid $dir_ref_vid_simple --mode "simp_ucf" >> $dir_save/metrics.log  2>&1
#fvd_ucf for stdit
CUDA_VISIBLE_DEVICES='3' python3 fvd.py --dir_videos $dir_videos --dir_results $dir_results --dir_ref_vid $dir_ref_vid_simple/0 --mode "simp_ucf" >> $dir_save/metrics.log  2>&1

#fvd_fp16 for latte
#CUDA_VISIBLE_DEVICES='2' python3 fvd.py --dir_videos $dir_videos --dir_results $dir_results --dir_ref_vid $dir_ref_vid_fp_latte --mode "fpfvd" >> $dir_save/metrics.log  2>&1
#fvd_fp16 for stdit
rm $dir_ref_vid_fp_stdit/metrics.log
CUDA_VISIBLE_DEVICES='3' python3 fvd.py --dir_videos $dir_videos --dir_results $dir_results --dir_ref_vid $dir_ref_vid_fp_stdit --mode "fpfvd" >> $dir_save/metrics.log  2>&1

# Clip-temp & Clip-sim
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 clip_score.py --dir_videos $dir_videos --dir_prompts $dir_prompts --dir_results $dir_results --metric 'clip_temp_score' >> $dir_save/metrics.log  2>&1
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 clip_score.py --dir_videos $dir_videos --dir_prompts $dir_prompts --dir_results $dir_results --metric 'clip_score' >> $dir_save/metrics.log  2>&1

# VQA_A and VQA_T(VQ) $$$
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 evaluate_a_set_of_videos.py --dir_videos $dir_videos --dir_results $dir_results >> $dir_save/metrics.log  2>&1
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 evaluate_a_set_of_videos.py --dir_videos $dir_videos --dir_results $dir_results

# Flow-Score (TC)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 optical_flow_scores.py --dir_videos $dir_videos --metric 'flow_score' --dir_results $dir_results >> $dir_save/metrics.log  2>&1

# IS(VQ)
#cd $EC_path
#cd ./metrics
#CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 is.py --dir_videos $dir_videos --dir_results $dir_results >> $dir_save/metrics.log  2>&1

# temporal flickering(TC)
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 temporal_flickering.py --dir_videos $dir_videos >> $dir_save/metrics.log  2>&1

# Dump cleaned output
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python read_metric_log.py  $dir_save/metrics.log >> $dir_videos/metrics.log 2>&1
CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python read_metric_log.py  $dir_save/metrics.log
