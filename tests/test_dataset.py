from tests.fixtures import mock_ab_dataframe, aho_sequence  # noqa: F401
from walkjump.data import AbDataset
from walkjump.constants import TOKENS_AHO


def test_abdataset(mock_ab_dataframe):  # noqa: F811
    dataset = AbDataset(mock_ab_dataframe, TOKENS_AHO)
    assert len(dataset[0]) == mock_ab_dataframe.loc[0].str.len().sum()
