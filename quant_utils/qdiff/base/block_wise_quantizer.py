import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time
import math

from qdiff.base.base_quantizer import BaseQuantizer, DynamicQuantizer

logger = logging.getLogger(__name__)

class BlockWiseAttnMapDynamicQuantizer(BaseQuantizer):
    """
    the input shape should be [Group,-1]
    store the quant params (delta, zp) offline with init_quant_params
    """

    def __init__(self, quant_config):
        super().__init__(quant_config)

    def quantize(self, x:torch.Tensor):
        '''
        TODO: some special techniques for attn_map quant
        '''
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
