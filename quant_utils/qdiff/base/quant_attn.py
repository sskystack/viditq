import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer
from qdiff.base.mixed_precision_quantizer import MixedPrecisionStaticQuantizer, MixedPrecisionDynamicQuantizer
from omegaconf import OmegaConf, ListConfig

class QuantizedAttentionMap(torch.nn.Module):
    """
    the quantization for attention map
    """
    def __init__(
        self,
        quant_config: dict,
    ) -> None:
        super().__init__()
        
        self.quant_config = quant_config
        self.group = self.quant_config.attn.attn_map.group  # [column, block]
        self.attn_map_quantizer = DynamicQuantizer(quant_config['attn']['attn_map'])
        reorder_file = self.quant_config.attn.qk.reorder_file_path
        self.optimal_reorder = torch.load(reorder_file, weights_only=True)
        if self.quant_config.attn.attn_map.get('int8_scale', False):
            dummy_int8_quant_config = OmegaConf.create({
                'n_bits': 8,
                'sym': True,
            })
            self.attn_map_scale_quantizer = DynamicQuantizer(dummy_int8_quant_config)
            
        self.mixed_precision_cfg = None
        self.i_block = None
        self.split_range = None   # choose a subset of heads due to memory issue.
        
        self.quant_mode = True   # when set as False, use the original model forward
        

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        input shape: [B,N_token,C]
        """
        if not self.quant_mode:  # use the FP
            return x
        else:
            BS, head_per_split_num, N_token, N_token = x.shape
            device = x.device
            if self.group == 'column':
                x = x.permute([0,1,3,2]).reshape([-1, N_token])    # a row shares same quant_params
                x_quant = self.attn_map_quantizer(x).reshape([BS, head_per_split_num, N_token, N_token]).permute([0,1,3,2])
                return x_quant
            elif self.group == 'block':
                BS, N_head, N_token, N_token = x.shape
                N_text_token = self.quant_config.model.n_text_tokens
                N_image_token = N_token - self.quant_config.model.n_text_tokens
                F = 13
                H = 30
                W = 45
                assert N_image_token == F*W*H
                
                # the text-text selfattn, text-image crossattn remain FP.
                attn_map_image = x[:,:,N_text_token:,N_text_token:]
                
                # generate quant_params, parallel quant forward
                # 1. get block-wise max for each head
                delta = torch.zeros([BS,N_head,N_image_token,N_image_token], device=device)
                if self.mixed_precision_cfg is not None:
                    mixed_precision = torch.zeros([BS,N_head,N_image_token,N_image_token], device=device)
                for i_bs in range(BS):  # bs=2
                    for i_head in range(N_head):  # N_head=48
                        i_order = self.optimal_reorder['permute_order_index'][self.i_block][i_head]
                        if self.quant_config.attn.attn_map.level_2:
                            num_block_per_dim = self.optimal_reorder['chunk_num_table'][i_order]*self.optimal_reorder['chunk_num_table_level_2'][i_order]
                        else:
                            num_block_per_dim = self.optimal_reorder['chunk_num_table'][i_order]
                        block_width_per_dim = N_image_token // num_block_per_dim
                        assert N_image_token % num_block_per_dim == 0, "block_size should be divisible by image token length"
                        
                        attn_map_image_head = attn_map_image[i_bs,i_head,:,:]  # [N_image_token, N_image_token]
                        attn_map_image_head = attn_map_image_head.unfold(0,block_width_per_dim,block_width_per_dim).unfold(1,block_width_per_dim,block_width_per_dim)
                        attn_map_image_head = attn_map_image_head.reshape([num_block_per_dim,num_block_per_dim,-1])
                        
                        delta_ = attn_map_image_head.max(dim=-1)[0]  # [block_size, block_size]
                                                
                        # INFO: get the int8: delta_int8 = quant(delta)
                        if self.quant_config.attn.attn_map.get('int8_scale', False):
                            delta_before_quant = delta_.clone()
                            delta_max = torch.zeros_like(delta_).fill_(delta_.max())
                            delta_ = self.attn_map_scale_quantizer.forward_with_quant_params(
                                delta_,
                                delta_max
                            )
                            # print((delta_before_quant - delta_)/delta_)
                        
                        if self.mixed_precision_cfg is not None:
                            assert self.quant_config.attn.attn_map.level_2, "mixed precision cfg file is associated with level-2 fine-grained block currently."
                            mixed_precision_ = self.mixed_precision_cfg[self.i_block,self.split_range,:]
                            assert mixed_precision_.shape == delta_.shape
                            mixed_precision_ = mixed_precision_.reshape([num_block_per_dim, num_block_per_dim,1,1]).repeat(1,1,block_width_per_dim,block_width_per_dim)
                            mixed_precision_ = mixed_precision_.permute([0,2,1,3]).reshape([N_image_token, N_image_token])
                            mixed_precision[i_bs,i_head,:,:] = mixed_precision_
                            
                        delta_ = delta_.reshape([num_block_per_dim, num_block_per_dim,1,1]).repeat(1,1,block_width_per_dim,block_width_per_dim)
                        delta_ = delta_.permute([0,2,1,3]).reshape([N_image_token, N_image_token])
                        delta[i_bs,i_head,:,:] = delta_
                        
                if self.mixed_precision_cfg is not None:
                    attn_map_image_quant = self.attn_map_quantizer.forward_with_quant_params(attn_map_image, delta, mixed_precision=mixed_precision)
                else:
                    attn_map_image_quant = self.attn_map_quantizer.forward_with_quant_params(attn_map_image, delta)
                x[:,:,N_text_token:,N_text_token:] = attn_map_image_quant
                
                return x
                
                # INFO: check quant result
                # print(((attn_map_image_quant-attn_map_image)/attn_map_image).max())
                
if __name__ == '__main__':
    dummy_q_config = {
        'weight': {
            'n_bits': 8,
            'group': 'tensor',
            'sym': False
        },
        'act': {
            'n_bits': 8,
            'group': 'tensor',
            'sym': False
        }
    }
    dummy_linear = nn.Linear(8,32, device='cuda')
    dummy_q_linear = QuantizedLinear(
        in_features = dummy_linear.in_features,
        out_features = dummy_linear.out_features,
        bias = dummy_linear.bias is not None,
        device = dummy_linear.weight.device,
        quant_config = dummy_q_config,
        fp_module = dummy_linear,
    )
    dummy_input = torch.rand([4,2048,8], device='cuda')
    output = dummy_q_linear(dummy_input)
