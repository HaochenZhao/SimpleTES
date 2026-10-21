# EVOLVE-BLOCK-START
from typing import Tuple, Dict
import torch
import triton
import triton.language as tl

@triton.jit
def _broken_kernel(x_ptr, n):
    offs = tl.arange(0, 16)
    tl.store(x_ptr + offs, offs)

def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]) -> torch.Tensor:
    x, mask, weights, config = data
    _broken_kernel[(1,)](x, x.numel())
    return x
# EVOLVE-BLOCK-END
