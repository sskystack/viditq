"""
follow the orginial model inference script, 
and add the following parts. 
to store the activation offline for activation probing and quantization
"""
import torch.nn as nn

from qdiff.utils import apply_hook_to_submodules
from qdiff.base.quant_layer import QuantizedLinear

class SaveActivationHook:

    def __init__(self):
        self.hook_handle = None
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        # self.outputs.append(module_in[0].abs().max(dim=1)[0])
        self.outputs.append(module_in[0])

    def clear(self):
        self.outputs = []

def add_hook_to_module_(module, hook_cls):
    hook = hook_cls()
    hook.hook_handle = module.register_forward_hook(hook)
    return hook

# import the quantized version of the model
# from model import QuantizedOpenSORA

# convert original model to quantized version
# def convert_model_quantized():
#     return qnn

# add the hook for output 
def add_hooks_for_calib_data():
    # INFO: need to reimplement for each model (or each feature to probe)
    pass

# save the calib data from hooked inputs
def save_calib_data():
    pass

if __name__ == '__main__':
    import torch

    # Define a simple 3-layer linear network
    class DummyNet(nn.Module):
        def __init__(self):
            super(DummyNet, self).__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 30)
            self.fc3 = nn.Linear(30, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Create an instance of the network
    net = DummyNet()
    net = net.to('cuda')

    import yaml
    yaml_file_path = '../config.yaml'
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)

    kwargs = {
        'hook_cls': SaveActivationHook,
    }

    hook_d = apply_hook_to_submodules(net,
                            class_type=nn.Linear,
                            hook_function=add_hook_to_module_,
                            has_return=True,
                            **kwargs
                            )
    dummpy_input = torch.rand([1,32,10], device='cuda')
    out = net(dummpy_input)

    for k,v in hook_d.items():
        print(f'layer_name: {k}, hook_input_shape: {v.outputs[0].shape}')
        v.hook_handle.remove()  # remove the hooks

    import ipdb; ipdb.set_trace()

    


