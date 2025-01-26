export HF_ENDPOINT=https://hf-mirror.com
GPU_ID="5"

CUDA_VISIBLE_DEVICES=$GPU_ID python get_calib_data.py configs/software_simulation.py \
 --prompt-path ./t2v_samples.txt
