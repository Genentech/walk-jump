from tests.fixtures import aho_sequence, mock_ab_dataframe  # noqa: F401
from walkjump.constants import TOKENS_AHO
from walkjump.data import AbDataset


def test_abdataset(mock_ab_dataframe):  # noqa: F811
    dataset = AbDataset(mock_ab_dataframe, TOKENS_AHO)
    print(dataset)
    assert len(dataset[0]) == mock_ab_dataframe.loc[0].str.len().sum()
