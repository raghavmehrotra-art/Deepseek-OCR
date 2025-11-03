import torch
from typing import List, Tuple


def tile_image(img: torch.Tensor, tile_hw: Tuple[int, int]) -> List[torch.Tensor]:
    return [img]
