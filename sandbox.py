if __name__ == "__main__":
    from tests.test_training_debug import test_training_debug, test_training_debug_distributed

    test_training_debug_distributed(
        arg_values=
        ["--num-layers", "2", "--hidden-size", "768", "--num-attention-heads", "12", "--micro-batch-size", "2",
         "--encoder-seq-length", "2048", "--max-position-embeddings", "2048", "--vocab-file",
         "/home/aleph/repos/samuel/bigscience_megatron_deepspeed/tests/data/gpt2/gpt2-tiny-vocab.json", "--merge-file",
         "/home/aleph/repos/samuel/bigscience_megatron_deepspeed/tests/data/gpt2/gpt2-tiny-merges.txt", "--data-path",
         "/home/aleph/repos/samuel/bigscience_megatron_deepspeed/tests/data/gpt2/meg-gpt2-openwebtext_text_document",
         "--split", "40,30,30", "--train-iters", "10", "--lr", "0.0001", "--min-lr", "0.00001", "--bf16",
         "--reset-attention-mask", "--no-masked-softmax-fusion", "--deepspeed", "--zero-stage", "0",
         "--deepspeed_config", "/home/aleph/repos/samuel/bigscience_megatron_deepspeed/tests/ds_config_zero_0.json",
         "--eval-iters", "-1", "--tensor-model-parallel-size", "2"
         ]
    )
