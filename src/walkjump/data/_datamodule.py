from lightning.pytorch import LightningDataModule
import pandas as pd
from dataclasses import dataclass, field
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder
from typing import Literal

from walkjump.constants import TOKENS_AHO
from ._dataset import AbDataset
from ._batch import AbBatch


@dataclass
class AbDataModule(LightningDataModule):
    csv_data_path: str
    batch_size: int = 64
    num_workers: int = 1

    dataset: pd.DataFrame = field(init=False)
    alphabet: LabelEncoder = field(init=False)

    def setup(self, stage: str):
        match stage:
            case "fit" | "validate" | "test":
                self.dataset = pd.read_csv(self.csv_data_path, compression="gzip")
                self.alphabet = LabelEncoder().fit(TOKENS_AHO)
            case _:
                raise ValueError(f"Unreognized 'stage': {stage}")

    def _make_dataloader(self, partition: Literal["train", "val", "test"]) -> DataLoader:
        df = self.dataset[self.dataset.partition == partition]
        dataset = AbDataset(df, self.alphabet)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=partition == "train",
            num_workers=self.num_workers,
            collate_fn=AbBatch.from_tensor_pylist,
        )

    def train_dataloader(self) -> DataLoader:
        return self._make_dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._make_dataloader("val")

    def test_dataloader(self) -> DataLoader:
        return self._make_dataloader("test")
