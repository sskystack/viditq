from opensora.models.stdit.stdit3 import STDiT3, STDiT3Config
from opensora.utils.ckpt_utils import load_checkpoint
from opensora.models.layers.blocks import (
    Attention,
    MultiHeadCrossAttention,
    LlamaRMSNorm,
)
import xformers.ops
from einops import rearrange

import torch
import sys
import os
import argparse
import numpy as np
from omegaconf import OmegaConf

import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer, BaseQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules
from qdiff.base.quant_model import quant_layer_refactor_, bitwidth_refactor_, load_quant_param_dict_, save_quant_param_dict_, set_init_done_
from qdiff.base.quant_attn import QuantizedAttentionMapOpenSORA

import logging
logger = logging.getLogger(__name__)

class QuantOpenSora(STDiT3):
    def __init__(
        self,
        quant_config:dict,
        config,
        from_pretrained
    ): 
        super().__init__(config)
        load_checkpoint(self, from_pretrained)

        self.quant_config=quant_config
        self.quant_param_dict = {}
        self.quant_layer_refactor()

    def convert_quant(self, quant_config):
        self.quant_config = quant_config
            
        self.quant_param_dict = {}
        self.quant_layer_refactor()
    
    def quant_layer_refactor(self):
        if self.quant_config.get('attn', False):
            # replace the attn layers, should be first, since Attention contains Linear layers
            apply_func_to_submodules(self, 
                    class_type=Attention,
                    function=quant_attn_refactor_,
                    name=None,
                    parent_module=None,
                    quant_config=self.quant_config,
                    full_name=None,
                    remain_fp_regex=self.quant_config.remain_fp_regex,
                    )
        if self.quant_config.get('cross_attn', False):
            # replace the attn layers, should be first, since Attention contains Linear layers
            apply_func_to_submodules(self, 
                    class_type=MultiHeadCrossAttention,
                    function=quant_cross_attn_refactor_,
                    name=None,
                    parent_module=None,
                    quant_config=self.quant_config,
                    full_name=None,
                    remain_fp_regex=self.quant_config.remain_fp_regex,
                    )
        # replace the linear layers
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
                model=self
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
        
    def bitwidth_refactor(self):
        apply_func_to_submodules(self,
                class_type=QuantizedLinear,
                function=bitwidth_refactor_,
                name=None,
                parent_module=None,
                quant_config=self.quant_config,
                full_name=None
                )
        
# -------------- for quant attention -----------

def quant_attn_refactor_(submodule,name,parent_module,quant_config,full_name,remain_fp_regex,class_type=None):
    
    quant_layer_type = QuantizedAttention      

    # set some layers as FP (fixed), feed in from config
    if remain_fp_regex is not None:
        import re
        pattern = re.compile(remain_fp_regex)
        if pattern.search(full_name):
            logger.info(f"remain {full_name} quant as FP due to fp_regex")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(parent_module, name, quant_layer_type(
            dim = submodule.dim,
            num_heads = submodule.num_heads,
            qkv_bias = submodule.qkv.bias is not None,
            qk_norm = not isinstance(submodule.q_norm, nn.Identity),
            attn_drop = submodule.attn_drop.p,
            proj_drop = submodule.proj_drop.p,
            # norm_layer:  use default
            enable_flash_attn = False, # force as False
            rope = submodule.rotary_emb if hasattr(submodule,'rotary_emb') else None, 
            qk_norm_legacy = submodule.qk_norm_legacy,
            quant_config=quant_config
        )
    )
    # set the qkv with parameters to new module (cause the load_checkpoint happens before this refactor)
    setattr(getattr(parent_module,name),'qkv',submodule.qkv)
    setattr(getattr(parent_module,name),'q_norm',submodule.q_norm)
    setattr(getattr(parent_module,name),'k_norm',submodule.k_norm)
    setattr(getattr(parent_module,name),'proj',submodule.proj)
    
    # specify temporal or spatial attn
    # parent module is the STDiTBlock
    setattr(getattr(parent_module,name),'temporal', parent_module.temporal)
    
    # set the module_name for quant_layer and quantizers
    setattr(getattr(parent_module, name), 'module_name', full_name)
    if getattr(parent_module, name).attn_map_quantizer is not None:
        setattr(getattr(parent_module, name).attn_map_quantizer, 'module_name', full_name)
    
def quant_cross_attn_refactor_(submodule,name,parent_module,quant_config,full_name,remain_fp_regex,class_type=None):
    
    quant_layer_type = QuantizedMultiHeadCrossAttention      

    # set some layers as FP (fixed), feed in from config
    if remain_fp_regex is not None:
        import re
        pattern = re.compile(remain_fp_regex)
        if pattern.search(full_name):
            logger.info(f"remain {full_name} quant as FP due to fp_regex")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setattr(parent_module, name, quant_layer_type(
            d_model = submodule.d_model,
            num_heads = submodule.num_heads,
            attn_drop = submodule.attn_drop.p,
            proj_drop = submodule.proj_drop.p,
            quant_config=quant_config
        )
    )
    # set the qkv with parameters to new module (cause the load_checkpoint happens before this refactor)
    setattr(getattr(parent_module,name),'q_linear',submodule.q_linear)
    setattr(getattr(parent_module,name),'kv_linear',submodule.kv_linear)
    setattr(getattr(parent_module,name),'proj',submodule.proj)
        
    # set the module_name for quant_layer and quantizers
    setattr(getattr(parent_module, name), 'module_name', full_name)
    if getattr(parent_module, name).attn_map_quantizer is not None:
        setattr(getattr(parent_module, name).attn_map_quantizer, 'module_name', full_name)

class QuantizedAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
        quant_config=None,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope
        
        self.is_causal = False
        
        # ------------- For Quantization ---------------
        # enable to turn on the intermediate savings 
        self.apply_hooks = False  
        if self.apply_hooks:
            self.hooks = {}  # hook_handle to store activations
        
        # the quant part.
        assert quant_config is not None
        self.quant_config = quant_config

        self.q_quantizer = None
        self.k_quantizer = None
        self.v_quantizer = None
        self.attn_map_quantizer = None
        
        if self.quant_config.attn.get('qk', None) is not None:
            self.q_quantizer = DynamicQuantizer(self.quant_config.attn.qk)
            self.k_quantizer = DynamicQuantizer(self.quant_config.attn.qk)
            if self.apply_hooks:
                self.hooks['q'] = add_hook_to_module_(self.q_quantizer, SaveActivationHook, type='qk')
                self.hooks['k'] = add_hook_to_module_(self.k_quantizer, SaveActivationHook, type='qk')
        
        if self.quant_config.attn.get('v', None) is not None:
            self.v_quantizer = DynamicQuantizer(self.quant_config.attn.v)
            if self.apply_hooks:
                self.hooks['v'] = add_hook_to_module_(self.v_quantizer, SaveActivationHook, type='v')
                
        if self.quant_config.attn.get('attn_map', None) is not None:
            self.attn_map_quantizer = QuantizedAttentionMapOpenSORA(self.quant_config)

            if self.apply_hooks:
                self.hooks['attn_map'] = add_hook_to_module_(self.attn_map_quantizer, SaveActivationHook, type='attn')        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
        
        '''INFO: (optionally) apply reorder.'''
        if self.quant_config.attn.get('qk', None) is not None:
            if self.quant_config.attn.qk.get('reorder', False):
                q, k, v = self.reorder_qk(q, k, v)
                
        '''INFO: Quantization of QKV'''
        if self.temporal:
            # shape: [288, 16, 30, 72]
            BS, N_head, N_temporal_token, N_dim = q.shape
            N_spatial_token = BS//2
            assert N_head == self.num_heads
            N_token = N_temporal_token
        else: # spatial_attn
            # shape: [60, 16, 144, 72]
            # DIRTY: regard the inference as BS=1, the we have 2 batch bacuse of CFG
            BS, N_head, N_spatial_token, N_dim = q.shape
            N_temporal_token = BS//2
            assert N_head == self.num_heads
            N_token = N_spatial_token
            
        if self.q_quantizer is not None:
            if self.apply_hooks:
                self.hooks['q'].original_shape = [BS, N_head, N_token, N_dim]
            q = self.q_quantizer(q.reshape([-1,N_dim])).reshape([BS, N_head, N_token, N_dim])
        if self.k_quantizer is not None:
            if self.apply_hooks:
                self.hooks['k'].original_shape = [BS, N_head, N_token, N_dim]
            k = self.k_quantizer(k.reshape([-1,N_dim])).reshape([BS, N_head, N_token, N_dim])
        if self.v_quantizer is not None:
            if self.apply_hooks:
                self.hooks['v'].original_shape = [BS, N_head, N_token, N_dim]
            v = self.v_quantizer(
                v.permute([0,1,3,2]).reshape([-1, N_token])  # all tokens share the same quant_params.
                ).reshape([BS, N_head, N_dim, N_token]).permute([0,1,3,2])

        if enable_flash_attn:
            raise AssertionError("quantized attention are not supported with flash_attn yet.")
            # from flash_attn import flash_attn_func
            # # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            # q = q.permute(0, 2, 1, 3)
            # k = k.permute(0, 2, 1, 3)
            # v = v.permute(0, 2, 1, 3)
            # x = flash_attn_func(
            #     q,
            #     k,
            #     v,
            #     dropout_p=self.attn_drop.p if self.training else 0.0,
            #     softmax_scale=self.scale,
            #     causal=self.is_causal,
            # )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            if self.is_causal:
                causal_mask = torch.tril(torch.ones_like(attn), diagonal=0)
                causal_mask = torch.where(causal_mask.bool(), 0, float('-inf'))
                attn += causal_mask
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            '''INFO: Quantization of the post softmax attn_map'''
            # shape: spatial: torch.Size([60, 16, 144, 144])
            assert attn.shape == (BS, N_head, N_token, N_token)
            if self.apply_hooks:
                self.hooks['attn_map'].original_shape = [BS, N_head, N_token, N_token]
            if self.attn_map_quantizer is not None:
                attn = self.attn_map_quantizer(attn)            
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class QuantizedMultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, quant_config=None):
        super(QuantizedMultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # ------------- For Quantization ---------------
        # enable to turn on the intermediate savings 
        self.apply_hooks = False  
        if self.apply_hooks:
            self.hooks = {}  # hook_handle to store activations
        
        # the quant part.
        assert quant_config is not None
        self.quant_config = quant_config

        self.q_quantizer = None
        self.k_quantizer = None
        self.v_quantizer = None
        self.attn_map_quantizer = None
        
        if self.quant_config.cross_attn.get('q', None) is not None:
            self.q_quantizer = DynamicQuantizer(self.quant_config.cross_attn.q)
            if self.apply_hooks:
                self.hooks['q'] = add_hook_to_module_(self.q_quantizer, SaveActivationHook, type='qk')
        
        if self.quant_config.cross_attn.get('kv', None) is not None:
            self.v_quantizer = DynamicQuantizer(self.quant_config.cross_attn.kv)
            self.k_quantizer = DynamicQuantizer(self.quant_config.cross_attn.kv)
            if self.apply_hooks:
                self.hooks['k'] = add_hook_to_module_(self.k_quantizer, SaveActivationHook, type='qk')
                self.hooks['v'] = add_hook_to_module_(self.v_quantizer, SaveActivationHook, type='v')
                
        if self.quant_config.cross_attn.get('attn_map', None) is not None:
            self.attn_map_quantizer = QuantizedAttentionMapOpenSORA(self.quant_config, cross_attn=True)

            if self.apply_hooks:
                self.hooks['attn_map'] = add_hook_to_module_(self.attn_map_quantizer, SaveActivationHook, type='attn')        

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        
        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        
        def pytorch_memory_efficient_attention(q, k, v, p=0.0, attn_bias=None):
            # Use equivalent python code from https://github.com/facebookresearch/xformers/blob/main/xformers/ops/fmha/__init__.py to replace the xformer attention.
            scale = 1.0 / q.shape[-1] ** 0.5
            q = q * scale
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            
            '''INFO: Quantization of the QKV'''
            BS, N_head, N_image_token, N_dim = q.shape
            BS, N_head, N_text_token, N_dim = v.shape
            
            if self.q_quantizer is not None:
                q_shape = q.shape
                if self.apply_hooks:
                    self.hooks['q'].original_shape = q_shape
                q = self.q_quantizer(q.reshape([-1,N_dim])).reshape(q_shape)
            if self.k_quantizer is not None:
                k_shape = k.shape
                if self.apply_hooks:
                    self.hooks['k'].original_shape = k_shape
                k = self.k_quantizer(k.reshape([-1,N_dim])).reshape(k_shape)
            if self.v_quantizer is not None:
                v_shape = v.shape
                if self.apply_hooks:
                    self.hooks['v'].original_shape = v_shape
                v = self.v_quantizer(
                    v.permute([0,1,3,2]).reshape([-1, N_text_token])  # all tokens share the same quant_params.
                    ).reshape([BS, N_head, N_dim, N_text_token]).permute([0,1,3,2])
            
            attn = q @ k.transpose(-2, -1).contiguous()
            dtype_ = attn.dtype
            attn.to(torch.float32)
            if attn_bias is not None:
                attn = attn + attn_bias.materialize(attn.shape).to(attn.device).to(attn.dtype)
            attn = attn.softmax(-1)
            attn = F.dropout(attn, self.attn_drop.p)
            attn.to(dtype_)
            
            '''INFO: Quantization of the Attention Map'''
            assert attn.shape == (BS, N_head, N_image_token, N_text_token)
            if self.apply_hooks:
                self.hooks['attn_map'].original_shape = (BS, N_head, N_image_token, N_text_token)
            if self.attn_map_quantizer is not None:
                attn = self.attn_map_quantizer(attn)     

            attn = attn @ v 
            attn = attn.transpose(1, 2).contiguous()
            return attn
        
        attn = pytorch_memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias) 

        # INFO: There is still like 0.014 maximum difference. and half the elments are not the same. With higher data precision (FP32), the max error is reduced to 0.001. but the output have no visual difference, so pass it on. 
        # attn_kernel = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)  # torch.Size([1, 8640, 16, 72])
        # print('Maximum Difference: ',(attn - attn_kernel).abs().max())
        # print('Different Elements Ratio: ',(attn != attn_kernel).sum() / attn.numel())

        x = attn
        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SaveActivationHook:

    def __init__(self, type=None, original_shape=None):
        self.hook_handle = None
        self.type = type
        self.original_shape = original_shape
        self.outputs = []
        self.attn_ds_rate = 64
        
    def attn_map_downsample(self, data):
        '''
        down_sample in the N_token dimension, handle the indivisible situation. 
        '''
        assert self.type == 'attn'
        BS, head_per_split_num, N_token, N_token = self.original_shape
        N_remainder = N_token % self.attn_ds_rate
        data = data[:,:,:-N_remainder,:-N_remainder]
        data = data.reshape([
            BS,head_per_split_num,N_token//self.attn_ds_rate,self.attn_ds_rate,N_token//self.attn_ds_rate,self.attn_ds_rate
            ])
        return data.max(dim=3)[0].max(dim=4)[0]
        
    def __call__(self, module, module_in, module_out):
        '''
        the input shape could be [BS, N_group];
        reduce along the head dimension. 
        '''
        if self.type == 'qk':
            BS, head_per_split_num, N_token, N_dim = self.original_shape
            data = module_in[0].reshape(self.original_shape).to('cpu')
            # data = module_in[0].reshape(self.original_shape).abs().max(dim=-1)[0].to('cpu') # avoid taking up too much GPU memory
        elif self.type == 'v':
            BS, head_per_split_num, N_dim, N_token = self.original_shape
            data = module_in[0].reshape(self.original_shape).to('cpu')
        elif self.type == 'attn':
            BS, head_per_split_num, N_token, N_token = self.original_shape
            data = module_in[0].reshape(self.original_shape).to('cpu')
            if self.attn_ds_rate is not None:
                data = self.attn_map_downsample(data)
        else:
            raise NotImplementedError
        
        # TODO: add processing. 
        self.outputs.append(data)

    def clear(self):
        self.outputs = []

def add_hook_to_module_(module, hook_cls, **kwargs):
    hook = hook_cls(**kwargs)
    hook.hook_handle = module.register_forward_hook(hook)
    return hook
