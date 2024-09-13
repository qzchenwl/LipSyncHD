import os
import sys
from typing import Optional, Union

import torch
import torch.nn as nn
import torchvision

WORK_DIR = os.path.dirname(__file__)
for x in ["DH_live", "GFPGAN"]:
    sys.path.insert(0, os.path.abspath(os.path.join(WORK_DIR, "libs", x)))

##################################################
# patch torchvision.transforms.functional_tensor #
##################################################
sys.modules['torchvision.transforms.functional_tensor'] = torchvision.transforms.functional

#####################
# patch torch.load #
####################
original_load = torch.load


def patched_load(*args, **kwargs):
    if 'map_location' not in kwargs and not torch.cuda.is_available():
        kwargs['map_location'] = 'cpu'
    return original_load(*args, **kwargs)


if not torch.cuda.is_available():
    torch.load = patched_load

########################
# patch nn.Module.cuda #
########################
# 保存原始的 cuda 方法
original_module_cuda = nn.Module.cuda


def patched_module_cuda(self: nn.Module, device: Optional[Union[int, torch.device]] = None) -> nn.Module:
    if torch.cuda.is_available():
        return original_module_cuda(self, device)
    else:
        return self.cpu()


if not torch.cuda.is_available():
    nn.Module.cuda = patched_module_cuda
