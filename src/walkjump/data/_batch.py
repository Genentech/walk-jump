from dataclasses import dataclass, field
from functools import cached_property

import torch
from walkjump.constants import TOKENS_AHO


@dataclass
class Batch:
    batch_tensor: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    tokens: list[str] = field(default_factory=lambda: TOKENS_AHO)

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], token_list: list[str] = TOKENS_AHO
    ) -> "Batch":

        packed_batch = torch.stack(inputs, dim=0)
        return cls(packed_batch, token_list)

    @cached_property
    def x(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()
