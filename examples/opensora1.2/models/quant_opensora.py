from opensora.models.stdit.stdit3 import STDiT3, STDiT3Config
from opensora.utils.ckpt_utils import load_checkpoint

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

# def quant_layer_refactor_(submodule,name,parent_module,quant_config,full_name):
#     if 'embedder' in full_name or 'adaLN_modulation' in full_name or 't_block' in full_name:
#         return
#     in_features=submodule.in_features
#     out_features=submodule.out_features
#     if submodule.bias is not None:
#         bias=True
#     else:
#         bias=False
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     setattr(parent_module, name, QuantizedLinear(in_features,out_features,bias,device,quant_config,submodule))
#     # also merge set_module_name_for_quantizer here. after replacing the quant_layer, also set the quantizer.

# def load_quant_param_dict_(submodule, full_name, quant_param_dict, model):
#     submodule.delta = quant_param_dict[full_name]['delta']
#     submodule.zero_point = quant_param_dict[full_name]['zero_point']

#     # update the quant_model.quant_param_dict also
#     model.quant_param_dict[full_name] = quant_param_dict[full_name]

# def save_quant_param_dict_(submodule, full_name, model):
#     model.quant_param_dict[full_name] = {}
#     model.quant_param_dict[full_name]['delta'] = submodule.delta
#     model.quant_param_dict[full_name]['zero_point'] = submodule.zero_point

# def set_init_done_(submodule):
#     submodule.init_done = True

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