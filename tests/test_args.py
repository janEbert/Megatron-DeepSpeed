import pytest
from argparse import Namespace

from megatron.arguments import parse_args

@pytest.mark.cpu
@pytest.mark.parametrize("arg_values,expected_error", [
    [["--num-layers", "2", "--hidden-size", "768", "--num-attention-heads", "12", "--micro-batch-size", "2", "--encoder-seq-length", "2048", "--max-position-embeddings", "2048"], None],
    [[], AssertionError],
])
def test_args(arg_values, expected_error):
    if expected_error is None:
        args = parse_args(args=arg_values)
        assert isinstance(args, Namespace)
    else:
        with pytest.raises(expected_error):
            args = parse_args(args=arg_values)