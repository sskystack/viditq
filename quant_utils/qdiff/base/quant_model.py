import torch
import torch.nn as nn
import torch.nn.functional as F
from qdiff.base.base_quantizer import StaticQuantizer, DynamicQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules

def quant_layer_refactor_():
    pass  # TODO:
    # also merge set_module_name_for_quantizer here. after replacing the quant_layer, also set the quantizer.

def load_quant_param_dict_():
    pass

def save_quant_param_dict_():
    pass

def set_init_done_():
    pass

class QuantModel(nn.Module):
    """
    the base quant model.
    specialized funcs should be implememted in subclass.
    (e.g., QuantizedOpenSORA...)
    """
    def __init__(
        self,
        quant_config: dict,
        **kwargs,
    ) -> None:
        super().__init__() # initialize all attributes from parent class

        # additional attributes for quant
        self.q_cfg = quant_config
        self.quant_param_dict = {}

        # refactor layers with quant_layers based on q_cfg
        self.quant_layer_refactor()
    
    def quant_layer_refactor(self):
        apply_func_to_submodules(self, 
                class_type=nn.Linear,
                function=quant_layer_refactor_)

    def save_quant_params_dict(self):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=save_quant_param_dict_)

    def load_quant_params_dict(self, quant_param_dict):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=load_quant_param_dict_,
                quant_param_dict=quant_param_dict)

    def set_init_done(self):
        apply_func_to_submodules(self, 
                class_type=BaseQuantizer,
                function=set_init_done_,)

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError("should be implemented in subclass.")

if __name__ == '__main__':
    # TODO: 
