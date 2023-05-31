import pytest
from walkjump.constants import TOKENS_AHO
from sklearn.preprocessing import LabelEncoder
import pandas as pd


@pytest.fixture(scope="session")
def aho_alphabet_encoder() -> LabelEncoder:
    return LabelEncoder().fit(TOKENS_AHO)


@pytest.fixture(scope="session")
def aho_sequence() -> str:
    return "EIVLTQSPATLSLSPGERATLSCRAS--QSVS------TYLAWYQQKPGRAPRLLIYD--------ASNRATGIPARFSGSGSG--TDFTLTISSLEPEDFAVYYCQQRSN------------------------WWTFGQGTKVEIK"  # noqa: E501


@pytest.fixture(scope="session")
def mock_ab_dataframe(aho_sequence) -> pd.DataFrame:
    return pd.DataFrame([{"fv_heavy_aho": aho_sequence, "fv_light_aho": aho_sequence}] * 100)
