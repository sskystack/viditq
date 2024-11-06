#CFG_NAME="config"
#EXP_NAME="cuda_test"

CFG_NAME="mixed_precision"
EXP_NAME="mixed_precision_test"

# quant infer command
CUDA_VISIBLE_DEVICES=1 python quant_inference.py \
    --image-size 256\
    --seed 1 \
    --ptq-config "./configs/${CFG_NAME}.yaml"\
	--log "./logs/${EXP_NAME}" \
	#--hardware \
	#--profile \

