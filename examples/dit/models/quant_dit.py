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
from models.models import DiT,DiT_models
from models.download import find_model
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
import argparse
import numpy as np
from omegaconf import OmegaConf

import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules
# save the calib data from hooked inputs
def load_calib_data():
    pass

# save the calib data from hooked inputs
# def save_quant_ckpt(loaded_model):
    # torch.save(loaded_model.state_dict(), 'model_params.pth')

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

def load_quant_param_dict_(submodule, full_name, quant_param_dict, model):
    submodule.delta = quant_param_dict[full_name]['delta']
    submodule.zero_point = quant_param_dict[full_name]['zero_point']

    # update the quant_model.quant_param_dict also
    model.quant_param_dict[full_name] = quant_param_dict[full_name]

def save_quant_param_dict_(submodule, full_name, model):
    model.quant_param_dict[full_name] = {}
    model.quant_param_dict[full_name]['delta'] = submodule.delta
    model.quant_param_dict[full_name]['zero_point'] = submodule.zero_point

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
        self.quant_param_dict = {}
        self.quant_layer_refactor()
    
    def quant_layer_refactor(self):
        apply_func_to_submodules(self, 
                class_type=nn.Linear,
                function=quant_layer_refactor_,
                name=None,
                parent_module=None,
                quant_config=self.quant_config,
                full_name=None
                )

    def save_quant_param_dict(self):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=save_quant_param_dict_,
                full_name=None,
                model=self
                )

    def load_quant_param_dict(self, quant_param_dict):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=load_quant_param_dict_,
                full_name=None,
                quant_param_dict=quant_param_dict,
                model=self,
                )

    def set_init_done(self):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=set_init_done_,)


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
