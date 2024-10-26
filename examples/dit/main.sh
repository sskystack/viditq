# python get_calib_data.py

# PTQ command
python ptq.py \
    --image-size 256\
    --seed 1 \
    --ptq-config './configs/sq.yaml'\
    #--ckpt '/home/zhuhongyu/DiT-main/pretrained_models/DiT-XL-2-512x512.pt'A

python quant_inference.py \
    --image-size 256\
    --seed 1 \
    --ptq-config './configs/sq.yaml'\

