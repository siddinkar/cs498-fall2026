"""Allocate checkpoint-backed parameters on meta, preserving destination dtypes.

Nonpersistent buffers still initialize normally. Strict assignment materializes
all registered tensors, and remaining meta tensors fail before inference.
"""
from contextlib import contextmanager
import torch
@contextmanager
def empty_parameters():
    original=torch.nn.Module.register_parameter
    def register(module,name,parameter):
        original(module,name,parameter)
        if parameter is not None:
            if type(parameter) is not torch.nn.Parameter:raise TypeError('Unsupported parameter subclass in empty initialization')
            module._parameters[name]=torch.nn.Parameter(parameter.to(device='meta'),requires_grad=parameter.requires_grad)
    torch.nn.Module.register_parameter=register
    try:yield
    finally:torch.nn.Module.register_parameter=original

def assign_weights(model,state):
    expected=model.state_dict()
    converted={name:value.to(dtype=expected[name].dtype) if name in expected else value for name,value in state.items()}
    model.load_state_dict(converted,strict=True,assign=True)
    missing=[name for name,value in list(model.named_parameters())+list(model.named_buffers()) if value.is_meta]
    if missing:raise ValueError('Unmaterialized tensors: '+str(missing))
