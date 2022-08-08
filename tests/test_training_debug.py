import pytest
import os
import torch
from pathlib import Path
from pretrain_gpt import main



@pytest.mark.skipif(not torch.cuda.is_available(), reason="no cuda available")
@pytest.mark.parametrize("arg_values", [
    ["--num-layers", "2", "--hidden-size", "768", "--num-attention-heads", "12", "--micro-batch-size", "2", "--encoder-seq-length", "2048", "--max-position-embeddings", "2048", "--vocab-file", str(Path(__file__).parent.absolute() / "data/gpt2/gpt2-tiny-vocab.json"), "--merge-file", str(Path(__file__).parent.absolute() / "data/gpt2/gpt2-tiny-merges.txt"), "--data-path", str(Path(__file__).parent.absolute() /"data/gpt2/meg-gpt2-openwebtext_text_document"), "--split", "40,30,30", "--train-iters", "10", "--lr", "0.0001", "--min-lr", "0.00001", "--bf16", "--reset-attention-mask", "--no-masked-softmax-fusion", "--deepspeed", "--zero-stage", "0", "--deepspeed_config", str(Path(__file__).parent.absolute() /"ds_config_zero_0.json"), "--eval-iters", "-1"]
])
def test_training_debug(arg_values):
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29001"
    os.environ["LOCAL_RANK"] = "0"
    main(args=arg_values)
    pass