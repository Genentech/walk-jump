import pytest
from walkjump.constants import TOKENS_AHO
from sklearn.preprocessing import LabelEncoder


@pytest.fixture(scope="session")
def aho_alphabet_encoder() -> LabelEncoder:
    return LabelEncoder().fit(TOKENS_AHO)
