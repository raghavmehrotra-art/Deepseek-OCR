import torch
import torch.nn as nn


class DummyMoE(nn.Module):
    def __init__(self, d_model: int = 1024, n_experts: int = 4):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = 0
        for e in self.experts:
            out = out + e(x)
        return out / len(self.experts)
