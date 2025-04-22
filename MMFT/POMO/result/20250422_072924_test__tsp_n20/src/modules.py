import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """From https://github.com/meta-llama/llama-models"""

    def __init__(self, dim: int, eps: float = 1e-5, **kwargs):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class ParallelGatedMLP(nn.Module):
    """From https://github.com/togethercomputer/stripedhyena"""

    def __init__(
        self,
        hidden_size: int = 128,
        inner_size_multiple_of: int = 256,
        mlp_activation: str = "silu",
        model_parallel_size: int = 1,
    ):
        super().__init__()

        multiple_of = inner_size_multiple_of
        self.act_type = mlp_activation
        if self.act_type == "gelu":
            self.act = F.gelu
        elif self.act_type == "silu":
            self.act = F.silu
        else:
            raise NotImplementedError

        self.multiple_of = multiple_of * model_parallel_size

        inner_size = int(2 * hidden_size * 4 / 3)
        inner_size = self.multiple_of * (
            (inner_size + self.multiple_of - 1) // self.multiple_of
        )

        self.l1 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l2 = nn.Linear(
            in_features=hidden_size,
            out_features=inner_size,
            bias=False,
        )
        self.l3 = nn.Linear(
            in_features=inner_size,
            out_features=hidden_size,
            bias=False,
        )

    def forward(self, z):
        z1, z2 = self.l1(z), self.l2(z)
        return self.l3(self.act(z1) * z2)