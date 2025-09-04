# Llama4 Fine-tuning with ms-swift

This folder contains an example of using [ms-swift](https://github.com/modelscope/ms-swift) to run multimodal fine-tuning for Llama4 OCR.

We will fine-tune Llama-4-Scout-17B-16E-Instruct on the [linxy/LaTeX_OCR](https://huggingface.co/datasets/linxy/LaTeX_OCR) dataset and provide the format for custom datasets.

The required GPU memory resources for this case are 4 * 65GiB, and it can be completed within 3 hours.

## Prerequisites

To run the following example, we need to install ms-swift:

```bash
pip install ms-swift transformers -U
```

## Training

The script for image OCR fine-tuning is as follows:
```shell
# GPU memory consumption: 4 * 65GiB
NPROC_PER_NODE=4 \
USE_HF=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model meta-llama/Llama-4-Scout-17B-16E-Instruct \
    --dataset 'linxy/LaTeX_OCR:full#5000' \
    --split_dataset_ratio 0.01 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_regex '^(language_model).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$' \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --deepspeed zero3 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4
```

If you want to use a custom dataset, simply specify it as follows:
```shell
    --dataset train.jsonl \
    --val_dataset val.jsonl \
```

The format of the custom dataset is as follows:
```jsonl
{"messages": [{"role": "user", "content": "Where is the capital of Zhejiang?"}, {"role": "assistant", "content": "The capital of Zhejiang is Hangzhou."}]}
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<image>Describe the image"}, {"role": "assistant", "content": "The image shows a little boy reading a book attentively by the window."}], "images": ["/xxx/x.png"]}
{"messages": [{"role": "user", "content": "<image><image>What is the difference between the two images?"}, {"role": "assistant", "content": "The first one is a kitten, and the second one is a puppy."}], "images": ["/xxx/y.jpg", "/xxx/z.png"]}
```


## Inference After Fine-tuning

Use the CLI to perform inference on the validation set:
```shell
# GPU memory consumption: 4 * 65GiB
USE_HF=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift infer \
    --adapters output/x-xxx/checkpoint-xxx \
    --stream false \
    --max_batch_size 4 \
    --load_data_args true \
    --max_new_tokens 512
```

- The `--adapters output/vx-xxx/checkpoint-xxx` here needs to be replaced with the folder of the last checkpoint generated during training. If using full-parameter training, it should be set to `--model <checkpoint-dir>`.
- The adapters folder contains the training parameter file `args.json`, so there is no need to additionally set `--model meta-llama/Llama-4-Scout-17B-16E-Instruct`.
- We specified `--split_dataset_ratio 0.01` during training, which splits 1% of the data as the validation set. You can use the validation set from training for inference by setting `--load_data_args true`. Alternatively, you can use `--val_dataset custom_val.jsonl` for inference.


Inference using transformers/peft:
```python
# GPU memory consumption: 4 * 60GiB
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['USE_HF'] = '1'

def get_messages():
    from datasets import load_dataset
    dataset = load_dataset('linxy/LaTeX_OCR', name='full', split='test')
    data_sample = dataset[0]
    image = data_sample['image']
    labels = data_sample['text']
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": "Using LaTeX to perform OCR on the image."},
            ]
        },
    ]
    return messages, labels

from transformers import Llama4ForConditionalGeneration, AutoProcessor
from peft import PeftModel
from modelscope import snapshot_download
model_dir = snapshot_download('meta-llama/Llama-4-Scout-17B-16E-Instruct')
adapter_dir = 'output/vx-xxx/checkpoint-xxx'
model = Llama4ForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype='auto', device_map='auto')
model = PeftModel.from_pretrained(model, adapter_dir)
processor = AutoProcessor.from_pretrained(model_dir)

messages, labels = get_messages()
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
]

response = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f'response: {response}')
print(f'labels: {labels}')
# response: \alpha _ { 1 } ^ { r } \gamma _ { 1 } + \ldots + \alpha _ { N } ^ { r } \gamma _ { N } = 0 \qquad ( r = 1 , \ldots , R ) ,
# label: \alpha _ { 1 } ^ { r } \gamma _ { 1 } + \dots + \alpha _ { N } ^ { r } \gamma _ { N } = 0 \quad ( r = 1 , . . . , R ) ,
```

Merge LoRA using CLI:
```shell
# GPU memory consumption: 4 * 55GiB
USE_HF=1 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift export \
    --adapters output/vx-xxx/checkpoint-xxx \
    --merge_lora true
```
