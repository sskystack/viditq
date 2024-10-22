"""
conduct model PTQ process
take in the orginal model and the calib data
save the quantized model checkpoint
"""
import torch
import sys
import os
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from models import DiT,DiT_models
from download import find_model
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import argparse
import numpy as np
from omegaconf import OmegaConf
from models.models import DiT,DiT_models
import torch.nn as nn
import torch.nn.functional as F
from quant_utils.qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from quant_utils.qdiff.base.quant_layer import QuantizedLinear
from quant_utils.qdiff.utils import apply_hook_to_submodules
from models.quant_dit import QuantDit
def main(args):

    # PTQ main function:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device="cuda" if torch.cuda.is_available() else "cpu"

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
    latent_size = args.image_size // 8
    ptq_config_file = args.ptq_config
    quant_config = OmegaConf.load(ptq_config_file)

    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    model=QuantDit(quant_config,
     ckpt_path,
     depth=28,
     hidden_size=1152, 
     patch_size=2, 
     num_heads=16, 
     input_size=latent_size,
     num_classes=args.num_classes,
     ).to(device)


    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    class_labels = [217, 363, 347, 574, 188, 99, 47, 379]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    t = torch.tensor([1] * z.shape[0], device=device)
    _=model(z,t,y)
    model.set_init_done()
    model.save_quant_params_dict()
    np.save('quant_params_dict.npy', model.quant_params_dict)
    # convert_model_quantized

    # conduct model inference

    # save the quant params
    save_quant_ckpt(model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument('--ptq-config', default='./configs/w8a8.yaml', type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)