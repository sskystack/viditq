from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm
from diffusers.models.attention import _chunked_feed_forward, GatedSelfAttentionDense

import xformers.ops
from viditq_extension.nn.base import QuantParams
from viditq_extension.nn.qlinear import W8A8OF16LinearDynamicInputScale
from viditq_extension.nn.layernorm import LayerNormGeneral
import viditq_extension.fused as fused_kernels

logger = logging.get_logger(__name__)

class BasicTransformerBlockWithCudaKernel(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        # INFO: used for cuda kernel. 
        quant_params: QuantParams = None,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        
        # INFO: add quant_params as module attribute.
        self.quant_params = quant_params

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None
            
        self.use_kernel = [True, True, True]   # [self_attn, cross_attn, ffn]

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)
            
        if self.use_kernel[0]:
            # INFO: replace attn. 
            self.attn1 = QuantAttentionWithCudaKernel(
                dim=dim,
                num_heads=num_attention_heads,
                quant_params=quant_params,
                has_bias=attention_bias,  # qkv mapping whether has bias, default True
                weight_sym=False,         # whether adopt symmetric weight quant
            )
            self.norm1 = LayerNormGeneral(dim, act_sum=True, eps=1e-6)
        else:
            self.attn1 = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )


        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            
            if self.use_kernel[1]:
                # INFO: replace attn. 
                self.attn2 = QuantCrossAttentionWithCudaKernel(
                    dim=dim,
                    num_heads=num_attention_heads,
                    quant_params=quant_params,
                    has_bias=attention_bias,  # qkv mapping whether has bias, default True
                    weight_sym=False,         # whether adopt symmetric weight quant
                )

            else:
                self.attn2 = Attention(
                    query_dim=dim,
                    cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    dropout=dropout,
                    bias=attention_bias,
                    upcast_attention=upcast_attention,
                    out_bias=attention_out_bias,
                )  # is self-attn if encoder_hidden_states is none

        else:
            self.norm2 = None
            self.attn2 = None
    
        # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
        # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
        # the second cross attention block.
        if norm_type == "ada_norm":
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm2 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:  # INFO: "ada_norm_single" falls in this case
            self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

        # 3. Feed-forward
        # INFO: no norm3 for pixart, since norm_type == 'ada_norm_single'
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )
        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None
        
        if self.use_kernel[2]:
            # INFO: replace mlp. 
            self.ff = QuantMlpWithCudaKernel(
                in_features=dim,
                hidden_features=dim*4, # mult has default value of 4
                quant_params=quant_params,
                has_bias=ff_bias,
                weight_sym=False,
            )
            # INFO: the norm2 forward with cuda kernel are implemented with FFN forward.
            # 
            self.norm2 = LayerNormGeneral(dim, act_sum=True, eps=1e-6)
        else:
            self.ff = FeedForward(
                dim,
                dropout=dropout,
                activation_fn=activation_fn,
                final_dropout=final_dropout,
                inner_dim=ff_inner_dim,
                bias=ff_bias,
            )
        
        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")
            
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        if not self.use_kernel[0]:  # use the original FP inference without kernel
        
            batch_size = hidden_states.shape[0]
            assert self.norm_type == "ada_norm_single"
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.norm_type == "ada_norm_zero":
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm1(hidden_states)
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif self.norm_type == "ada_norm_single":
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1) 
                norm_hidden_states = self.norm1(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
                norm_hidden_states = norm_hidden_states.squeeze(1) 
            else:
                raise ValueError("Incorrect norm used")

            assert self.pos_embed is None
            if self.pos_embed is not None:
                norm_hidden_states = self.pos_embed(norm_hidden_states)
            
            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
            
            # # 1. Prepare GLIGEN inputs
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,  # only_cross_attention=False, being None
                attention_mask=attention_mask,   # None
                **cross_attention_kwargs,    # {}
            )
            if self.norm_type == "ada_norm_zero":
                attn_output = gate_msa.unsqueeze(1) * attn_output
            elif self.norm_type == "ada_norm_single":
                attn_output = gate_msa * attn_output
            hidden_states = attn_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
                
            # 1.2 GLIGEN Control
            if gligen_kwargs is not None:
                hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])
            
        else:

            cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
            gligen_kwargs = cross_attention_kwargs.pop("gligen", None)
            
            assert self.norm_type == "ada_norm_single"
            assert gligen_kwargs is None
            assert cross_attention_kwargs == {}
        
            # INFO: replace with cuda kernel, fuse norm with quant operator
            batch_size = hidden_states.shape[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                    self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).to(torch.float16).chunk(6, dim=1)  # INFO: add to torch.float16 because currently cuda kernel only support it. 
            hidden_states = hidden_states.to(torch.float16)
            hidden_states = hidden_states.contiguous()
            
            residual = hidden_states
            norm_hidden_states = self.norm1(hidden_states, shift_msa, scale_msa, self.quant_params)
            
            # INFO: the cuda kernel inference of attn.
            B, N, C  = norm_hidden_states.shape
            attn_out = self.attn1(norm_hidden_states)
            hidden_states = fused_kernels.gate_residual_fuse(attn_out.view(-1, C), gate_msa.view(-1, C), residual.view(-1, C)).view(B, N, C)

        # 3. Cross-Attention
        if self.attn2 is not None:
            
            if not self.use_kernel[1]:
                
                # INFO: for pixart models, we donot have norm2. 
                if self.norm_type == "ada_norm":
                    norm_hidden_states = self.norm2(hidden_states, timestep)
                elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                    norm_hidden_states = self.norm2(hidden_states)
                elif self.norm_type == "ada_norm_single":
                    # For PixArt norm2 isn't applied here:
                    # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                    norm_hidden_states = hidden_states
                elif self.norm_type == "ada_norm_continuous":
                    norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
                else:
                    raise ValueError("Incorrect norm")

                if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                    norm_hidden_states = self.pos_embed(norm_hidden_states)

                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states
            else:
                B, N, C  = hidden_states.shape
                residual = hidden_states
                hidden_states = fused_kernels.quant_sum(hidden_states, self.quant_params.sum_input, self.quant_params.scale_input)
                attn_out = self.attn2(hidden_states, encoder_hidden_states, encoder_attention_mask)
                hidden_states = residual + attn_out

        # 4. Feed-forward
        if not self.use_kernel[2]:
            # i2vgen doesn't have this norm 🤷‍♂️
            if self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
            elif not self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm3(hidden_states)

            if self.norm_type == "ada_norm_zero":
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            if self.norm_type == "ada_norm_single":
                norm_hidden_states = self.norm2(hidden_states)
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
            else:
                ff_output = self.ff(norm_hidden_states)

            if self.norm_type == "ada_norm_zero":
                ff_output = gate_mlp.unsqueeze(1) * ff_output
            elif self.norm_type == "ada_norm_single":
                ff_output = gate_mlp * ff_output

            hidden_states = ff_output + hidden_states
            if hidden_states.ndim == 4:
                hidden_states = hidden_states.squeeze(1)
                
        else:
            B, N, C  = hidden_states.shape
            residual = hidden_states
            norm_hidden_states = self.norm2(hidden_states, shift_mlp, scale_mlp, self.quant_params)
            ff_output = self.ff(norm_hidden_states)
            hidden_states = fused_kernels.gate_residual_fuse(ff_output.view(-1, C), gate_mlp.view(-1, C), residual.view(-1, C)).view(B, N, C)
        
        return hidden_states

class QuantAttentionWithCudaKernel(nn.Module):
    def __init__(
        self,
        dim,
        num_heads: int,
        quant_params: QuantParams,
        has_bias: bool = True,
        weight_sym: bool = True,
    ):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # self.qkv = W8A8OF16LinearDynamicInputScale(dim, dim * 3, has_bias=has_bias, weight_sym=weight_sym)\
        self.to_q = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.to_k = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.to_v = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        self.to_out = torch.nn.ModuleList([
            W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym),
            nn.Dropout(p=0.0)]
        )

        self.quant_params = quant_params

    def forward(self, x):
        B, N, C = x.shape
        
        q = self.to_q(x, self.quant_params)
        k = self.to_k(x, self.quant_params)
        v = self.to_v(x, self.quant_params)

        dtype = q.dtype

        q = q.view(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.view(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.view(B, N, self.num_heads, C // self.num_heads).to(dtype)

        x = xformers.ops.memory_efficient_attention(q, k, v).view(B, N, C)

        # # test
        # q = quantize(q, asym=False)
        # k = quantize(k, asym=True)
        # v = quantize(v, asym=True)
        # q = q.transpose(1, 2)
        # k = k.transpose(1, 2)
        # v = v.transpose(1, 2)
        # attn_map = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # # quantize attn_map
        # attn_map = torch.exp(attn_map - attn_map.max(dim=-1, keepdim=True).values)
        # attn_map = attn_map * 128
        # for i in range(attn_map.shape[0]):
        #     for j in range(attn_map.shape[1]):
        #         # if attn_map[i][j].min() > 8:
        #             attn_map[i][j] = torch.round(attn_map[i][j])

        # attn_map = attn_map / 128
        
        # attn_map = attn_map / attn_map.to(torch.float32).sum(dim=-1, keepdim=True).to(torch.float16)

        # x = (attn_map @ v).transpose(1, 2).reshape(B, N, C).contiguous()

        x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.to_out[0](x, self.quant_params)

        return x
    
class QuantCrossAttentionWithCudaKernel(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        quant_params: QuantParams,
        has_bias: bool = True,
        weight_sym: bool = True,
    ):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.to_q = W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym)
        # self.to_q = nn.Linear(dim, dim).to(torch.float16)
        self.to_k = nn.Linear(dim, dim).to(torch.float16)
        self.to_v = nn.Linear(dim, dim).to(torch.float16)
        self.to_out = torch.nn.ModuleList([
            W8A8OF16LinearDynamicInputScale(dim, dim, has_bias=has_bias, weight_sym=weight_sym),
            nn.Dropout(p=0.0)]   # DIRTY: original diffuser implementation has an empty dropout, in order to align the state_dict
        )

        self.quant_params = quant_params
    
    # copied from diffusers Attention class
    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.num_heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def forward(self, x, cond, mask=None):
        B, N, C = x.shape
        _, target_length, _x = cond.shape
        self.head_dim = C // self.num_heads
        assert C % self.num_heads == 0
        attn_mask = mask.unsqueeze(1).repeat(1,self.num_heads,1,1)  # [B, num_heads, 1, target_length]
        
        # if mask is not None:
        #     attn_bias = self.prepare_attention_mask(mask, target_length, B)  # [16, 1, 600]
        #     # expand our mask's singleton query_tokens dimension:
        #     #   [batch*heads,            1, key_tokens] ->
        #     #   [batch*heads, query_tokens, key_tokens]q
        #     # so that it can be added as a bias onto the attention scores that xformers computes:
        #     #   [batch*heads, query_tokens, key_tokens]
        #     # we do this explicitly because xformers doesn't broadcast the singleton dimension for us.
        #     attn_bias = attn_bias.expand(-1, N, -1)  # [16, 8192, 600]
        #     attn_bias = attn_bias.unsqueeze(0).reshape([B, self.num_heads, N, target_length])   # [1, 16, 8192, 600]
        # else:
        #     attn_bias = None
        
        q = self.to_q(x, self.quant_params).view(B, -1, self.num_heads, self.head_dim).transpose(1,2)  # [1, 8192, 16, 72]
        # q = self.to_q(x).reshape([B,N,self.num_heads,-1]).transpose(1,2) # [1, 8192, 16, 72]
        k = self.to_k(cond).reshape([B,target_length,self.num_heads,-1]).transpose(1,2)  # [1, 600, 16, 72]
        v = self.to_v(cond).reshape([B,target_length,self.num_heads,-1]).transpose(1,2)  # [1, 600, 16, 72]
        
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        x = x.transpose(1, 2).reshape(B, N, self.num_heads*self.head_dim)
        
        # INFO: xformer only support seq_len%8==0, 300/8 does not satisfy, so merge the batch dimension in. 
        # x = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=attn_bias).view(B, N, C)  # [1. 8192, 16, 72] -> [2, 4096, 1152]

        x = fused_kernels.quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.to_out[0](x, self.quant_params)

        return x

class QuantMlpWithCudaKernel(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        quant_params: QuantParams,
        has_bias: bool = True,
        weight_sym: bool = True,
    ):
        super().__init__()
        
        # INFO: align the diffuser's 
        self.net = nn.ModuleList(
            [
                nn.ModuleDict({
                    'proj': W8A8OF16LinearDynamicInputScale(in_features, hidden_features, has_bias=has_bias, weight_sym=weight_sym),  # fc1
                }),
                nn.Dropout(p=0.0),
                W8A8OF16LinearDynamicInputScale(hidden_features, in_features, has_bias=has_bias, weight_sym=weight_sym),  # fc2
            ]
        )
        # self.fc1 = W8A8OF16LinearDynamicInputScale(in_features, hidden_features, has_bias=has_bias, weight_sym=weight_sym),  # fc1
        # self.fc2 = W8A8OF16LinearDynamicInputScale(hidden_features, in_features, has_bias=has_bias, weight_sym=weight_sym),  # fc2
        self.quant_params = quant_params

    def forward(self, x):
        x = self.net[0].proj(x, self.quant_params)        
        x = fused_kernels.gelu_quant_sum(x, self.quant_params.sum_input, self.quant_params.scale_input)
        x = self.net[2](x, self.quant_params)
        return x
    
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