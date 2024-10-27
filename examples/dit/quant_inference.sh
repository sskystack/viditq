CFG_NAME="sq"
EXP_NAME="sq"

# quant infer command
python quant_inference.py \
    --image-size 256\
    --seed 1 \
    --ptq-config './configs/${CFG_NAME}.yaml'\
	--log './logs/${EXP_NAME}' \
    #--ckpt '/home/zhuhongyu/DiT-main/pretrained_models/DiT-XL-2-512x512.pt'
