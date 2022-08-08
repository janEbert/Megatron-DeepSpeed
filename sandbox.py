if __name__ == "__main__":
    from tests.test_args import test_args

    test_args(arg_values=["--num-layers", "2", "--hidden-size", "768", "--num-attention-heads", "12", "--micro-batch-size", "2", "--encoder-seq-length", "2048", "--max-position-embeddings", "2048"], expected_error=None)