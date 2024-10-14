"""
follow the orginial model inference script, 
and add the following parts. 
to store the activation offline for activation probing and quantization
"""

# import the quantized version of the model
from model import QuantizedOpenSORA

# convert original model to quantized version
def convert_model_quantized():
    return qnn

# add the hook for output 
def add_hooks_for_calib_data():
    pass

# save the calib data from hooked inputs
def save_calib_data():
    pass


