# Source: https://www.philschmid.de/fine-tune-llms-in-2024-with-trl

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import setup_chat_format, SFTTrainer

dataset = load_dataset(
    "json", data_files="train_text2sql_sft_dataset.json", split="train"
)

model_id = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.05,
    r=256,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

args = TrainingArguments(
    output_dir="llama31-8b-text2sql-epochs-20",  # directory to save and repository id
    num_train_epochs=20,  # number of training epochs
    per_device_train_batch_size=3,  # batch size per device during training
    gradient_accumulation_steps=2,  # number of steps before performing a backward/update pass
    gradient_checkpointing=True,  # use gradient checkpointing to save memory
    optim="adamw_torch_fused",  # use fused adamw optimizer
    logging_steps=10,  # log every 10 steps
    save_strategy="epoch",  # save checkpoint every epoch
    learning_rate=2e-4,  # learning rate, based on QLoRA paper
    bf16=True,  # use bfloat16 precision
    tf32=True,  # use tf32 precision
    max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
    warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
    lr_scheduler_type="constant",  # use constant learning rate scheduler
    push_to_hub=True,  # push model to hub
    report_to="tensorboard",  # report metrics to tensorboard
)

max_seq_length = 4096

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset,
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    peft_config=peft_config,
    packing=True,
)

trainer.train()

trainer.save_model()
