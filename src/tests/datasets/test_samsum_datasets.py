# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import pytest
from dataclasses import dataclass
from functools import partial
from unittest.mock import patch
from datasets import load_dataset

@dataclass
class Config:
    model_type: str = "llama"

try:
    load_dataset("knkarthick/samsum")
    SAMSUM_UNAVAILABLE = False
except ValueError:
    SAMSUM_UNAVAILABLE = True

@pytest.mark.skipif(SAMSUM_UNAVAILABLE, reason="Samsum dataset is unavailable")
@pytest.mark.skip_missing_tokenizer
@patch('llama_cookbook.finetuning.train')
@patch('llama_cookbook.finetuning.AutoTokenizer')
@patch("llama_cookbook.finetuning.AutoConfig.from_pretrained")
@patch("llama_cookbook.finetuning.AutoProcessor")
@patch("llama_cookbook.finetuning.MllamaForConditionalGeneration.from_pretrained")
@patch('llama_cookbook.finetuning.LlamaForCausalLM.from_pretrained')
@patch('llama_cookbook.finetuning.optim.AdamW')
@patch('llama_cookbook.finetuning.StepLR')
def test_samsum_dataset(
    step_lr,
    optimizer,
    get_model,
    get_mmodel,
    processor,
    get_config,
    tokenizer,
    train,
    mocker,
    setup_tokenizer,
    llama_version,
    ):
    from llama_cookbook.finetuning import main

    setup_tokenizer(tokenizer)
    get_model.return_value.get_input_embeddings.return_value.weight.shape = [32000 if "Llama-2" in llama_version else 128256]
    get_mmodel.return_value.get_input_embeddings.return_value.weight.shape = [0]
    get_config.return_value = Config()

    BATCH_SIZE = 8
    kwargs = {
        "model_name": llama_version,
        "batch_size_training": BATCH_SIZE,
        "val_batch_size": 1,
        "use_peft": False,
        "dataset": "samsum_dataset",
        "batching_strategy": "padding",
        }

    main(**kwargs)

    assert train.call_count == 1

    args, kwargs = train.call_args
    train_dataloader = args[1]
    eval_dataloader = args[2]
    token = args[3]

    VAL_SAMPLES = 818
    TRAIN_SAMPLES = 14732

    assert len(train_dataloader) == TRAIN_SAMPLES // BATCH_SIZE
    assert len(eval_dataloader) == VAL_SAMPLES

    batch = next(iter(train_dataloader))

    assert "labels" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "attention_mask" in batch.keys()

    assert batch["input_ids"][0][0] == token.bos_token_id
    assert batch["labels"][0][-1] == token.eos_token_id
    assert batch["input_ids"][0][-1] == token.eos_token_id
