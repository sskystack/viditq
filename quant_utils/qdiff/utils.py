class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

def apply_hook_to_submodules(module, class_type, hook_function, parent_name="", **kwargs):
    """
    Recursively iterates through all submodules of a PyTorch module and applies a hook function
    if the submodule matches the specified class type. The parent name is appended to the submodule name.

    Args:
        module (torch.nn.Module): The PyTorch module to iterate through.
        class_type (type): The class type to match against submodules.
        hook_function (callable): The function to apply if a submodule matches the class type.
        parent_name (str): The name of the parent module (used for recursion).
    """
    for name, submodule in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        if isinstance(submodule, class_type):
            hook_function(submodule, full_name, **kwargs)
        
        # Recursively apply the function to submodules
        apply_hook_to_submodules(submodule, class_type, hook_function, full_name)
