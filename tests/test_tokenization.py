from walkjump.utils import token_string_from_tensor, token_string_to_tensor

TEST_FV_LIGHT_AHO = "EIVLTQSPATLSLSPGERATLSCRAS--QSVS------TYLAWYQQKPGRAPRLLIYD--------ASNRATGIPARFSGSGSG--TDFTLTISSLEPEDFAVYYCQQRSN------------------------WWTFGQGTKVEIK"  # noqa: E501


def test_token_to_string_tofrom_tensor(aho_alphabet_encoder):
    assert (
        TEST_FV_LIGHT_AHO
        == token_string_from_tensor(
            token_string_to_tensor(TEST_FV_LIGHT_AHO, aho_alphabet_encoder),
            aho_alphabet_encoder,
            from_logits=False,
        )[0]
    )
    assert (
        TEST_FV_LIGHT_AHO
        == token_string_from_tensor(
            token_string_to_tensor(TEST_FV_LIGHT_AHO, aho_alphabet_encoder, onehot=True),
            aho_alphabet_encoder,
            from_logits=True,
        )[0]
    )
