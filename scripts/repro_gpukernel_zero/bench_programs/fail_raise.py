# EVOLVE-BLOCK-START
from typing import Tuple, Dict
import torch

def custom_kernel(data: Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict]) -> torch.Tensor:
    raise RuntimeError("bench_fail_raise")
# EVOLVE-BLOCK-END
