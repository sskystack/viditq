"""
conduct model PTQ process
take in the orginal model and the calib data
save the quantized model checkpoint
"""
import torch
import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from models.models import DiT,DiT_models
from download import find_model
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import argparse

from omegaconf import OmegaConf

import torch.nn as nn
import torch.nn.functional as F
from quant_utils.qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from quant_utils.qdiff.base.quant_layer import QuantizedLinear
from quant_utils.qdiff.utils import apply_hook_to_submodules
# save the calib data from hooked inputs
def load_calib_data():
    pass

# save the calib data from hooked inputs
def save_quant_ckpt():
    pass

def quant_layer_refactor_(submodule,name,parent_module,quant_config,full_name):
    if 't_embedder' in full_name or 'adaLN_modulation' in full_name:
        return
    in_features=submodule.in_features
    out_features=submodule.out_features
    if submodule.bias is not None:
        bias=True
    else:
        bias=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(parent_module, name, QuantizedLinear(in_features,out_features,bias,device,quant_config,submodule))
    # also merge set_module_name_for_quantizer here. after replacing the quant_layer, also set the quantizer.

def load_quant_param_dict_():
    pass

def save_quant_param_dict_(submodule, full_name,origin_module):
    origin_module.quant_params_dict[full_name] = []
    origin_module.quant_params_dict[full_name].append(submodule.delta)
    origin_module.quant_params_dict[full_name].append(submodule.zero_point)

def set_init_done_(submodule):
    submodule.init_done = True

class QuantDit(DiT):
    def __init__(
        self,
        quant_config:dict,
        ckpt_path,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        **kwargs
    ): 
        super().__init__(
        input_size,
        patch_size,
        in_channels,
        hidden_size,
        depth,
        num_heads,
        mlp_ratio,
        class_dropout_prob,
        num_classes,
        learn_sigma)
        state_dict = find_model(ckpt_path)
        self.quant_config=quant_config
        self.load_state_dict(state_dict)
        self.quant_params_dict = {}
        self.quant_layer_refactor()
    
    def quant_layer_refactor(self):
        apply_hook_to_submodules(self, 
                class_type=nn.Linear,
                hook_function=quant_layer_refactor_,
                name=None,
                parent_module=None,
                quant_config=self.quant_config,
                full_name=None
                )

    def save_quant_params_dict(self):
        apply_hook_to_submodules(self, 
                class_type=BaseQuantizer,
                hook_function=save_quant_param_dict_,
                full_name=None,
                origin_module=self
                )

    def load_quant_params_dict(self, quant_param_dict):
        apply_hook_to_submodules(self, 
                class_type=BaseQuantizer,
                hook_function=load_quant_param_dict_,
                quant_param_dict=quant_param_dict)

    def set_init_done(self):
        apply_hook_to_submodules(self, 
                class_type=BaseQuantizer,
                hook_function=set_init_done_,)

        
        


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
    print(model.quant_params_dict)
    # convert_model_quantized

    # conduct model inference

    # save the quant params
    save_quant_ckpt()

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