import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
import logging
import os
from typing import Optional

class GlobalLogger:
    _initialized = False
    _log_file = None

    @classmethod
    def initialize(cls, log_file):
        if not cls._initialized:
            cls._log_file = log_file
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            
            handlers = [logging.FileHandler(log_file), logging.StreamHandler()]
            for handler in handlers:
                handler.setFormatter(formatter)
                root_logger.addHandler(handler)
            
            cls._initialized = True
    
    @classmethod
    def reset_logger(cls):
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:  # Copy the list to avoid iteration issues
            root_logger.removeHandler(handler)
        cls._initialized = False

    @classmethod
    def get_logger(cls, name=None):
        return logging.getLogger(name)

def regular_set(model, paras=([],[],[])):
    for n, module in model._modules.items():
       
        if 'batchnorm' in module.__class__.__name__.lower():
            for name, para in module.named_parameters():
                paras[2].append(para)
                #print("paras[2]")
        elif len(list(module.children())) > 0:
            paras = regular_set(module, paras)
            #print("recursive")
        elif module.parameters() is not None:
            for name, para in module.named_parameters():
                paras[1].append(para)
                #print("paras[1]")
    return paras


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SeqToANNContainer(nn.Module):
    # This code is form spikingjelly https://github.com/fangwei123456/spikingjelly
    def __init__(self, *args):
        super().__init__()
        if len(args) == 1:
            self.module = args[0]
        else:
            self.module = nn.Sequential(*args)

    def forward(self, x_seq: torch.Tensor):
        
        y_shape = [x_seq.shape[0], x_seq.shape[1]]
        #print(x_seq.flatten(0, 1).contiguous().shape)
        y_seq = self.module(x_seq.flatten(0, 1).contiguous())
        y_shape.extend(y_seq.shape[1:])
        return y_seq.view(y_shape)

class tdLayer(nn.Module):
    def __init__(self, layer):
        super(tdLayer, self).__init__()
        self.layer = SeqToANNContainer(layer)
       

    def forward(self, x):
        x_ = self.layer(x)
      
        return x_

class tdBatchNorm(nn.Module):
    def __init__(self, out_panel):
        super(tdBatchNorm, self).__init__()
        self.bn = nn.BatchNorm2d(out_panel)
        self.seqbn = SeqToANNContainer(self.bn)

    def forward(self, x):
        y = self.seqbn(x)
        return y


def replace_layer_by_tdlayer(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_layer_by_tdlayer(module)
        if module.__class__.__name__ == 'Conv2d':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'Linear':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'BatchNorm2d':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'AvgPool2d':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'Flatten':
            model._modules[name] = nn.Flatten(start_dim=-3,end_dim=-1)
        if module.__class__.__name__ == 'Dropout':
            model._modules[name] = tdLayer(model._modules[name])
        if module.__class__.__name__ == 'AdaptiveAvgPool2d':
            model._modules[name] = tdLayer(model._modules[name])       
    return model

def isActivation(name):
    if 'spike_layer' in name.lower() :
        return True
    return False

def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model


def add_dimension(x, T):
    x.unsqueeze_(1)
    x = x.repeat(1, T, 1, 1, 1)
    return x

def isActivation_spike(name):
        if 'spike_layer' in name.lower():
            return True
        return False

def snn_to_ann(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = snn_to_ann(module)
        if module.__class__.__name__ == 'SPIKE_layer':
            model._modules[name] = nn.Linear(module.in_features, module.out_features)
            model._modules[name].weight = Parameter(module.weight)
            model._modules[name].bias = Parameter(module.bias)
        elif module.__class__.__name__ == 'tdLayer':
            model._modules[name] = module.layer.module
        elif module.__class__.__name__ == 'Flatten':
            model._modules[name] = nn.Flatten()
    return model


