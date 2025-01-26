export HF_ENDPOINT=https://hf-mirror.com
GPU_ID="0"

#CUDA_VISIBLE_DEVICES=$GPU_ID python get_calib_data.py configs/software_simulation.py 

CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py configs/software_simulation.py

CUDA_VISIBLE_DEVICES=$GPU_ID python quant_inference.py configs/software_simulation.py
