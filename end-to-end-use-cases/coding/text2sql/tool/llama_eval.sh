eval_path='./data/dev_20240627/dev.json'
db_root_path='./data/dev_20240627/dev_databases/'
ground_truth_path='./data/'

# Llama model on Hugging Face Hub
# https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# YOUR_API_KEY='huggingface'
# model='meta-llama/Llama-3.1-8B-Instruct'

# Fine-tuned Llama model locally
#YOUR_API_KEY='finetuned'
#model='fine_tuning/llama31-8b-text2sql-epochs-3'
#model='fine_tuning/llama31-8b-text2sql-epochs-8'

YOUR_API_KEY='xxx'
# Llama models on Together
#model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
#model='meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo'
model='meta-llama/Llama-3.3-70B-Instruct-Turbo'
#model='meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8'
#model='meta-llama/Llama-4-Scout-17B-16E-Instruct'

#YOUR_API_KEY='yyy'
# Llama models on Llama API
#model='Llama-3.3-8B-Instruct'
#model='Llama-3.3-70B-Instruct'
#model='Llama-4-Maverick-17B-128E-Instruct-FP8'
#model='Llama-4-Scout-17B-16E-Instruct-FP8'

#model="llama31-8b-text-sql-epochs-25"
#model="llama31-8b-text-sql-epochs-3"
#model="llama31-8b-text-sql"

data_output_path="./output/$model/v2/"

echo "Text2SQL using $model"
python3 -u llama_text2sql.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} \
--model ${model} --eval_path ${eval_path} --data_output_path ${data_output_path}

# Check if llama_text2sql.py exited successfully
if [ $? -eq 0 ]; then
    echo "llama_text2sql.py completed successfully. Proceeding with evaluation..."
    python3 -u text2sql_eval.py --db_root_path ${db_root_path} --predicted_sql_path ${data_output_path} \
    --ground_truth_path ${ground_truth_path} \
    --diff_json_path ${eval_path}

    echo "Done evaluating $model."

else
    echo "Error: llama_text2sql.py failed with exit code $?. Skipping evaluation."
    exit 1
fi
