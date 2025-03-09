import torch
from typing import Optional
from labsurv.builders import IMITATORS


@IMITATORS.register_module()
class BaseImitator:
    def __init__(self, device: Optional[str] = None):
        self.device = torch.cuda.device(device)