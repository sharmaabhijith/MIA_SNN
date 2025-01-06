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

