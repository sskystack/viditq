import torch
import os
import sys
import diffusers
import time
import shutil
import argparse
import logging

from qdiff.utils import apply_func_to_submodules, seed_everything, setup_logging
from omegaconf import OmegaConf, ListConfig

from models.customize_pipeline_pixart_sigma import CustomizePixArtSigmaPipeline
from models.customize_pipeline_pixart_transformer_2d import CustomizePixArtTransformer2DModel
# DIRTY: apply monkey patch, since the from_pretrained() method is hard to hack
diffusers.models.PixArtTransformer2DModel = CustomizePixArtTransformer2DModel
diffusers.PixArtSigmaPipeline = CustomizePixArtSigmaPipeline
from diffusers import PixArtSigmaPipeline

def main(args):
    seed_everything(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    if args.log is not None:
        if not os.path.exists(args.log):
            os.makedirs(args.log)
    log_file = os.path.join(args.log, 'run.log')
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    ckpt_path = args.ckpt if args.ckpt is not None else "./pretrained_models/"
    
    # INFO: donot convert_quant to acquire the original pipeline. 
    fp_pipe = PixArtSigmaPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16  # due to CUDA kernel only supports fp16, we donot use bfloat16 here. 
    ).to(device)
      
    quant_pipe = PixArtSigmaPipeline.from_pretrained(
        ckpt_path,
        torch_dtype=torch.float16  # due to CUDA kernel only supports fp16, we donot use bfloat16 here. 
    ).to(device)
    
    # ---- assign quant configs ------
    quant_config = OmegaConf.load(args.quant_config)
    quant_pipe.convert_quant(quant_config)
    model = quant_pipe.transformer
    
    if_mixed_precision = isinstance(quant_config.weight.n_bits, ListConfig) or isinstance(quant_config.act.n_bits, ListConfig)
    if if_mixed_precision:
        model.bitwidth_refactor()
        
    args.hardware = True
    if args.hardware:  # use the cuda kernel
        assert not if_mixed_precision, ("mixed precision is currently not supported in CUDA kernels")
        if args.quant_weight_ckpt is None:
            save_path = os.path.join(args.log, 'int_weight.pt')
            # INFO: always regenerate the int_weigjt
            quant_param_ckpt = torch.load(os.path.join(args.log, args.quant_param_ckpt), weights_only=True, map_location='cuda')
            model.load_quant_param_dict(quant_param_ckpt)
            model.quantize_and_save_weight(save_path=save_path)
            # if not os.path.exists(save_path):
            # else:
            #     logger.info('int_weight.pth exists, loading from the local file...')
            model.hardware_forward_refactor(load_path=save_path)

    else:  # use the algorithm simulation
        quant_param_ckpt = torch.load(os.path.join(args.log, args.quant_param_ckpt), weights_only=True)
        model.load_quant_param_dict(quant_param_ckpt)
    
    model.set_init_done()
    logger.info(str(model))

    # INFO: if memory intense
    # pipe.enable_model_cpu_offload()
    # pipe.vae.enable_tiling()
    
    # INFO: profile the latency. 
    torch.cuda.synchronize()
    fp_start = torch.cuda.Event(enable_timing=True)
    fp_end = torch.cuda.Event(enable_timing=True)
    quant_start = torch.cuda.Event(enable_timing=True)
    quant_end = torch.cuda.Event(enable_timing=True)
    
    # read the promts
    prompt_path = args.prompt if args.prompt is not None else "./prompts.txt"
    prompts = []
    with open(prompt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            prompts.append(line.strip())
        
    # INFO: record the FP time. 
    N_batch = len(prompts) // args.batch_size # drop_last
    for i in range(N_batch):
        # Warmup 1 Iter
        if i >= 1:
            fp_start.record()
        images = fp_pipe(
            prompt=prompts[i*args.batch_size: (i+1)*args.batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images
        print(f"Export image of batch {i}")
        save_path = os.path.join(args.log, "generated_images_fp")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i_image in range(args.batch_size):
            images[i_image].save(os.path.join(save_path, f"output_{i_image + args.batch_size*i}.jpg"))
    fp_end.record()
    torch.cuda.synchronize()
    fp_avg_time = fp_start.elapsed_time(fp_end) / (N_batch-1-1)
    logger.info(f"FP avg. inference time {fp_avg_time:.2f}ms")
    
    # INFO: record the quant time. 
    N_batch = len(prompts) // args.batch_size # drop_last
    for i in range(N_batch):
        # Warmup 1 Iter
        if i >= 1:
            quant_start.record()
        images = quant_pipe(
            prompt=prompts[i*args.batch_size: (i+1)*args.batch_size],
            num_inference_steps=args.num_sampling_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images
        print(f"Export image of batch {i}")
        save_path = os.path.join(args.log, "generated_images")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for i_image in range(args.batch_size):
            images[i_image].save(os.path.join(save_path, f"output_{i_image + args.batch_size*i}.jpg"))
    quant_end.record()
    torch.cuda.synchronize()
    quant_avg_time = quant_start.elapsed_time(quant_end) / (N_batch-1-1)
    logger.info(f"Quant avg. inference time {quant_avg_time:.2f}ms")
    
    logger.info("Quant E2E speedup: {:.3f}".format(fp_avg_time/quant_avg_time))
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str)
    parser.add_argument('--quant-config', required=True, type=str)
    parser.add_argument("--quant_param_ckpt", type=str, default="./quant_params.pth")
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=10)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action='store_true', help='profile mode, measure the e2e latency')
    parser.add_argument("--quant_weight_ckpt", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--ckpt", type=str, default=None)
    args = parser.parse_args()
    main(args)