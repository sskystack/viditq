import torch.nn as nn

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

def apply_func_to_submodules(module, class_type, function, parent_name="", has_return=False, **kwargs):
    """
    Recursively iterates through all submodules of a PyTorch module and applies a hook function
    if the submodule matches the specified class type. The parent name is appended to the submodule name.

    Args:
        module (torch.nn.Module): The PyTorch module to iterate through.
        class_type (type): The class type to match against submodules.
        function (callable): The function to apply if a submodule matches the class type.
        parent_name (str): The name of the parent module (used for recursion).
    """
    if has_return:
        return_d = {}

    for name, submodule in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        parent_module = module

        if 'name' in kwargs:
            kwargs['name']=name
        if 'full_name' in kwargs:
            kwargs['full_name'] = full_name
        if 'parent_module' in kwargs:
            kwargs['parent_module'] = module
        # if 'quant_param_dict' in kwargs:
            # kwargs['quant_param_dict'] = quant_param_dict
        if isinstance(submodule, class_type):
            if has_return:
                return_d[full_name] = function(submodule, **kwargs)
            else:
                function(submodule, **kwargs)

        # Recursively apply the function to submodules
        apply_func_to_submodules(submodule, class_type, function, full_name, **kwargs)
    if has_return:
        return return_d

