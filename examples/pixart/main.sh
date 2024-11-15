LOG='test'
CFG='config.yaml'
GPU_ID=0

CUDA_VISIBLE_DEVICES=$GPU_ID python ptq.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"
CUDA_VISIBLE_DEVICES=$GPU_ID python quant_inference.py --quant-config "./configs/${CFG}" --log "./logs/${LOG}"
