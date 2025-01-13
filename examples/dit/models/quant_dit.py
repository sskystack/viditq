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
import logging
from omegaconf import OmegaConf

import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.base.quant_model import quant_layer_refactor_, bitwidth_refactor_, load_quant_param_dict_, save_quant_param_dict_, set_init_done_
from qdiff.utils import apply_func_to_submodules

logger = logging.getLogger(__name__)

# ------- some layers with cuda kernel forward ---------
from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
from viditq_extension.nn.layernorm import LayerNormGeneral
import viditq_extension.fused as fused_kernels

def quantize_and_save_weight_(submodule, full_name):
    fp_weight = submodule.fp_module.weight.to(torch.float16)
    # the viditq_extension.nn.qlinear use [C] as the scale shape, but the qdiff simulation code use [C, 1]

    submodule.w_quantizer.delta = submodule.w_quantizer.delta.view(-1).to(torch.float16)
    submodule.w_quantizer.zero_point = submodule.w_quantizer.zero_point.view(-1).to(torch.float16)
    scale = submodule.w_quantizer.delta
    zero_point = submodule.w_quantizer.zero_point  # the cuda kernel code uses 128+zero_point

    # INFO: the orginal module weight is the FP16 quantized dequant weight, 
    # replace with INT weight, should update the state_dict
    int_weight = torch.clamp(
            torch.round(fp_weight / scale.view(-1,1)) - zero_point.view(-1,1),
            -128, 127).to(torch.int8)  # kernel supports W8A8 only for now
    submodule.weight.data = int_weight


from timm.models.layers import to_2tuple
class QuantMlpWithCudaKernel(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
        self, 
        in_features, 
        quant_params,
        hidden_features=None, 
        out_features=None, 
        act_layer=nn.GELU, 
        bias=True, 
        weight_sym=True,
        drop=0.,):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = W8A8OF16LinearDynamicInputScale(in_features, hidden_features, has_bias=bias, weight_sym=weight_sym)
        self.fc2 = W8A8OF16LinearDynamicInputScale(hidden_features, in_features, has_bias=bias, weight_sym=weight_sym)

        assert act_layer == nn.GELU  # support gelu only
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])

        self.quant_params = quant_params

    def forward(self, x):
        x = self.fc1(x, self.quant_params)
        x = fused_kernels.gelu_quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.drop1(x)
        x = self.fc2(x, self.quant_params)
        x = self.drop2(x)
        return x

class QuantAttentionWithCudaKernel(nn.Module):
    def __init__(
        self, 
        dim, 
        quant_params,
        num_heads=8,
        qkv_bias=False,
        weight_sym=True,
        attn_drop=0.,
        proj_drop=0.,
        ):
        super().__init__()

        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = W8A8OF16LinearDynamicInputScale(dim, dim * 3, has_bias=qkv_bias, weight_sym=weight_sym)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=True, weight_sym=weight_sym)
        self.proj_drop = nn.Dropout(proj_drop)

        self.quant_params = quant_params

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x, self.quant_params).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.proj(x, self.quant_params)
        x = self.proj_drop(x)
        return x

from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class QuantDiTBlockWithCudaKernel(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(
        self, 
        hidden_size, 
        num_heads, 
        quant_params,
        mlp_ratio=4.0, 
        **block_kwargs
        ):
        super().__init__()

        self.norm1 = LayerNormGeneral(hidden_size, act_sum=True, eps=1e-6)
        self.attn = QuantAttentionWithCudaKernel(
                hidden_size,
                quant_params=quant_params,
                num_heads=num_heads,
                qkv_bias=True,
                weight_sym=False,
            )
        self.norm2 = LayerNormGeneral(hidden_size, act_sum=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = QuantMlpWithCudaKernel(
                in_features=hidden_size,
                quant_params=quant_params,
                hidden_features=mlp_hidden_dim,
                bias=True,
                weight_sym=False,
                # act_layer=approx_gelu,  # kernel only support normal nn.GELU, set as default
                drop=0
        )
        self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.quant_params = quant_params

    def forward(self, x, c):

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        B, N, C = x.shape

        x = x.contiguous()
        residual = x
        x = self.norm1(x, shift_msa, scale_msa, self.quant_params)
        x = self.attn(x)
        B, N, C = x.shape
        x = fused_kernels.gate_residual_fuse(x.view(-1, C), gate_msa.view(-1, C), residual.view(-1, C)).view(B, N, C)

        residual = x
        x = self.norm2(x, shift_mlp, scale_mlp, self.quant_params)
        x = self.mlp(x)
        x = fused_kernels.gate_residual_fuse(x.view(-1, C), gate_mlp.view(-1, C), residual.view(-1, C)).view(B, N, C)

        return x

class QuantDiT(DiT):
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
                full_name=None,
                remain_fp_regex=self.quant_config.remain_fp_regex,
                )

    def save_quant_param_dict(self):
        apply_func_to_submodules(self,
                class_type=BaseQuantizer,
                function=save_quant_param_dict_,
                full_name=None,
                parent_module=None,
                model=self,
                )

    def load_quant_param_dict(self, quant_param_dict):
        apply_func_to_submodules(self,
                class_type=BaseQuantizer,
                function=load_quant_param_dict_,
                full_name=None,
                parent_module=None,
                quant_param_dict=quant_param_dict,
                model=self,
                )

    def set_init_done(self):
        apply_func_to_submodules(self,
                class_type=BaseQuantizer,
                function=set_init_done_,)

    # ------ used for infer with CUDA kernel ------- 

    def quantize_and_save_weight(self, save_path):

        # set require_grad=False, since torch force the variable to be FP or complex (we assign them as torch.int8)
        for param in self.parameters():
            param.requires_grad_(False)

        # iter through all QuantLayers and get quantized INT, fill into the state_dict
        apply_func_to_submodules(self,
                class_type=QuantizedLinear,
                function=quantize_and_save_weight_,
                full_name=None,
                )
        # delete the quant_params and fp_weights in the state_dict
        sd = self.state_dict()
        keys_to_delte = ['fp_weight','fp_module']
        keys_to_rename = {
                'w_quantizer.delta': 'scale_weight',
                'w_quantizer.zero_point': 'zp_weight',
                }
        for k in list(sd.keys()):
            if any(substring in k for substring in keys_to_delte):
                del sd[k]
            if any(substring in k for substring in keys_to_rename):
                original_k = k
                for substring in keys_to_rename.keys():
                    if substring in k:
                        k = k.replace(substring, keys_to_rename[substring])
                sd[k] = sd.pop(original_k)

        # IMPORTANT: modify the zero_point since cuda kernel use 128 as the base zero_point instead of 0.
        # FIXED in PTQ
        # for k in list(sd.keys()):
            # if 'zp_weight' in k:
                # sd[k] = (sd[k] + 128).to(torch.int16)

        # INFO: we implement the "general" version of layernorm
        # with the affine transform (wx+b) fused after the layernorm operation
        # so the vanilla layernorm has w=1 and b=0
        n_block = len(self.blocks)
        for i_block in range(n_block):
            hidden_size = self.blocks[i_block].mlp.fc1.in_features
            sd['blocks.{}.norm1.weight'.format(i_block)] = torch.ones((hidden_size,), dtype=torch.float16)
            sd['blocks.{}.norm2.weight'.format(i_block)] = torch.ones((hidden_size,), dtype=torch.float16)

        logger.info('Remaining Keys')
        for k in list(sd.keys()):
            logger.info('key: {},  {}'.format(k, sd[k].dtype))
        torch.save(sd, save_path)

        logger.info("\nFinished Saving the Quantized Checkpoint...\n")


    def hardware_forward_refactor(self, load_path):

        from viditq_extension.nn.base import QuantParams

        # (1) Set the seq_len to init the QuantParams 
        # the per-token activation quantization has token-wise quant_params
        # ----- ImageNet 256x256 -------
        # latent_size: 256//8 = 32
        # patch_emb: 2x2
        # token_len = 16x16=256
        # ------------------------------
        seq_len = 16 * 16 * 2 # 2*token_len, currently only support batch_size=1
        self.quant_params = QuantParams(seq_len, has_sum_input=True, device=torch.device("cuda"))

        # (2) replace the blocks, with cuda kernel version
        n_block = len(self.blocks)
        for i_block in range(n_block):
            block_ = self.blocks[i_block]
            hidden_size = block_.mlp.fc1.in_features
            num_heads = block_.attn.num_heads
            old_block = self.blocks[i_block]
            self.blocks[i_block] = QuantDiTBlockWithCudaKernel(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    # mlp_ratio = 4 by default
                    quant_params=self.quant_params,
                    ).half().to('cuda')
            setattr(self.blocks[i_block],'block_id',i_block)

        # (3) load the integer weights
        quant_sd = torch.load(load_path, weights_only=True, map_location='cuda')
        self.load_state_dict(quant_sd, strict=False)

    # ------ used for infer with mixed precision ------- 
    def bitwidth_refactor(self):
        apply_func_to_submodules(self,
                class_type=QuantizedLinear,
                function=bitwidth_refactor_,
                name=None,
                parent_module=None,
                quant_config=self.quant_config,
                full_name=None
                )


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
