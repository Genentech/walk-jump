from dataclasses import dataclass, InitVar, field
from functools import cached_property
from sklearn.preprocessing import LabelEncoder

import torch
from walkjump.constants import TOKENS_AHO


@dataclass
class Batch:
    batch_tensor: torch.Tensor
    """(b, L)-shaped tensor of sequences"""
    token_list: InitVar[list[str]] = TOKENS_AHO
    alphabet: LabelEncoder = field(init=False)

    def __post_init__(self, token_list: list[str]):
        self.alphabet = LabelEncoder().fit(token_list)
    
    @property
    def vocab_size(self) -> int:
        return len(self.alphabet.classes_)

    @classmethod
    def from_tensor_pylist(
        cls, inputs: list[torch.Tensor], token_list: list[str] = TOKENS_AHO
    ) -> "Batch":

        packed_batch = torch.stack(inputs, dim=0)
        return cls(packed_batch, token_list)
    
    @cached_property
    def x(self) -> torch.Tensor:
        return torch.nn.functional.one_hot(self.batch_tensor, num_classes=self.vocab_size).float()
