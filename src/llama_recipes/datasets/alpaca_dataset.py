# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import itertools
import json

import torch
from torch.utils.data import Dataset


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


class InstructionDataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.ann = json.load(open(dataset_config.data_path))
        # Use 5% of the dataset for evaluation
        eval_length = int(len(self.ann) / 20)
        if partition == "train":
            self.ann = self.ann[eval_length:]
        else:
            self.ann = self.ann[:eval_length]

        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]
        if ann.get("input", "") == "":
            user_prompt = PROMPT_DICT["prompt_no_input"].format_map(ann)
        else:
            user_prompt = PROMPT_DICT["prompt_input"].format_map(ann)

        # If vocab size is above 128000, use the chat template to generate the tokens as it is from Llama 3 family models
        if self.tokenizer.vocab_size >= 128000:
            dialog = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": ann["output"]},
            ]
            dialog_tokens = self.tokenizer.apply_chat_template(dialog)
            eot_indices = [i for i, n in enumerate(dialog_tokens) if n == 128009]
            labels = copy.copy(dialog_tokens)
            last_idx = 0
            # system prompt header "<|start_header_id|>system<|end_header_id|>" has been tokenized to [128006, 9125, 128007]
            # user prompt header "<|start_header_id|>user<|end_header_id|>" has been tokenized to [128006, 882, 128007]
            prompt_header_seqs = [[128006, 9125, 128007], [128006, 882, 128007]]
            for n, idx in enumerate(eot_indices):
                current_seq = labels[last_idx : idx + 1]
                if self.check_header(prompt_header_seqs, current_seq):
                    # found prompt header, indicating that this seq should be masked
                    labels[last_idx : idx + 1] = [IGNORE_INDEX] * (idx - last_idx + 1)
                else:
                    last_idx = idx + 1
            # Lastly mask all the assistant header prompt <|start_header_id|>assistant<|end_header_id|>, which has been tokenized to [128006, 78191, 128007]
            assistant_header_seq = [128006, 78191, 128007]
            labels = self.replace_target(assistant_header_seq, labels)
            dialog_tokens = [dialog_tokens]
            labels_tokens = [labels]
            combined_tokens = {
                "input_ids": list(itertools.chain(*(t for t in dialog_tokens))),
                "labels": list(itertools.chain(*(t for t in labels_tokens))),
            }
            return dict(
                combined_tokens, attention_mask=[1] * len(combined_tokens["input_ids"])
            )
        else:
            # for llama 2 fine-tuning, we use the old prompt template
            example = user_prompt + ann["output"]
            prompt = torch.tensor(self.tokenizer.encode(user_prompt), dtype=torch.int64)
            example = self.tokenizer.encode(example)
            example.append(self.tokenizer.eos_token_id)
            example = torch.tensor(example, dtype=torch.int64)
            labels = copy.deepcopy(example)
            labels[: len(prompt)] = -1
            example_mask = example.ge(0)
            label_mask = labels.ge(0)
            example[~example_mask] = 0
            labels[~label_mask] = IGNORE_INDEX

            return {
                "input_ids": example.tolist(),
                "labels": labels.tolist(),
                "attention_mask": example_mask.tolist(),
            }

    # check system prompt token seq or user prompt token seq is in the current token list
    def check_header(self, targets, seq):
        for i in range(len(seq) - 3):
            if seq[i : i + 3] in targets:
                return True
        return False

    def replace_target(self, target, seq):
        for i in range(len(seq) - 3):
            if seq[i : i + 3] == target:
                seq[i], seq[i + 1], seq[i + 2] = -100, -100, -100
        return seq
