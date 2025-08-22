import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

peft_model_path = "../fine-tuning/final_test/llama31-8b-text2sql-peft-nonquantized-cot"
output_dir = (
    "../fine-tuning/final_test/llama31-8b-text2sql-peft-nonquantized-cot_merged"
)
# === Load Base Model and Tokenizer ===
print("Loading base model and tokenizer...")
base_model_id = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Configure quantization if needed
quantization_config = None
use_quantized = False
if use_quantized:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

# Load model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=quantization_config,
)
base_model.resize_token_embeddings(128257)

# === Load PEFT Adapter and Merge ===
print("Loading PEFT adapter and merging...")
# peft_config = PeftConfig.from_pretrained(peft_model_path)
model = PeftModel.from_pretrained(base_model, peft_model_path)
model = model.merge_and_unload()  # This merges the adapter weights into the base model

# === Save the Merged Model ===
print(f"Saving merged model to {output_dir} ...")
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print("Done! The merged model is ready for vLLM serving.")
