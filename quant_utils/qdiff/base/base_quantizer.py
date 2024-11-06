import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time
import math

logger = logging.getLogger(__name__)

class BaseQuantizer(nn.Module):

    def __init__(self, quant_config):
        super(BaseQuantizer, self).__init__()
        
        # unpack the quant configurations
        self.n_bits = quant_config['n_bits']
        self.group = quant_config['group']
        self.sym = quant_config.get('sym', False)

        if isinstance(self.n_bits, list):
            raise AssertionError("when multiple n_bits are adopted, use the MixedPrecisionBaseQuantizer")
        assert self.group in ['token','tensor','channel']
        
        self.register_buffer('delta', None)
        self.register_buffer('zero_point', None)

        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.init_done = False
    
    def forward(self, x: torch.Tensor):
        raise NotImplementedError("should be implemented in subclass.")
    
    def init_quant_params(self, x):
        raise NotImplementedError("should be implemented in subclass.")

class StaticQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    store the quant params (delta, zp) offline with init_quant_params
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)

        if self.sym:
            self.abs_max = None
        else:
            self.x_max = None
            self.x_min = None
    
    def forward(self, x: torch.Tensor):
        x_quant = self.quantize(x)
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant
    
    def quantize(self, x: torch.Tensor):
    
        if self.init_done is not True:  # set as True in ptq.py
            self.init_quant_params(x)
        x_int = torch.round(x / self.delta) - self.zero_point
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        return x_quant
    
    def init_quant_params(self, x):

        assert len(x.shape) == 2  # [N_group, -1]

        if self.sym:
            x_absmax = x.abs().max(dim=1)[0]
            self.x_absmax = torch.max(self.x_absmax, x_absmax) if self.x_absmax is not None else x_absmax  # update

            delta = x_absmax / self.n_levels
            zero_point = torch.zeros_like(delta, device=delta.device)
        else:
            x_max = x.max(dim=1)[0]
            x_max[x_max<0] = 0. 
            # sometimes the weight are init on CPU, but new data on GPU needed for update quant_params (quarot)
            self.x_max = torch.max(self.x_max.to(x_max.device), x_max) if self.x_max is not None else x_max

            x_min = x.min(dim=1)[0]
            x_min[x_min>0] = 0.
            self.x_min = torch.min(self.x_min.to(x_min.device), x_min) if self.x_min is not None else x_min

            delta = (x_max - x_min)/(self.n_levels-1)
            zero_point = torch.round(x_min/delta) + (self.n_levels/2)
        
        assert torch.all(delta > 1.e-4), "unexpected small delta exists"

        self.delta = delta.unsqueeze(-1)  # [G] -> [G,1]
        self.zero_point = zero_point.unsqueeze(-1)

class DynamicQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    store the quant params (delta, zp) offline with init_quant_params
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)

    def quantize(self, x:torch.Tensor):
         # get the quant_params online
        assert len(x.shape) == 2  # [N_group, -1]

        if self.sym:
            x_absmax = x.abs().max(dim=1)[0]
            self.x_absmax = x_absmax

            delta = x_absmax / self.n_levels
            zero_point = torch.zeros_like(delta, device=delta.device)
        else:
            x_max = x.max(dim=1)[0]
            x_max[x_max<0] = 0. 
            self.x_max = x_max

            x_min = x.min(dim=1)[0]
            x_min[x_min>0] = 0.
            self.x_min = x_min

            delta = (x_max - x_min)/(self.n_levels-1)
            # INFO: check small values for delta, close to zero delta, would cause nan in zero_point
            eps = 1.e-4
            try:
                assert torch.all(delta.abs() > eps)
            except:
                import ipdb; ipdb.set_trace()
                delta[delta < eps] = eps
                logger.info("unexpected small delta: {:.3f} exists in {}, set as eps".format(delta.abs().min(), self.module_name))
            zero_point = torch.round(x_min/delta) + (self.n_levels/2)

        self.delta = delta.unsqueeze(-1)  # [G] -> [G,1]
        self.zero_point = zero_point.unsqueeze(-1)

        # quantize model with quant params
        x_int = torch.round(x / self.delta) - self.zero_point
        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        return x_quant

    def forward(self, x: torch.Tensor):
        x_quant = self.quantize(x)
        x_dequant = (x_quant + self.zero_point) * self.delta
        return x_dequant
