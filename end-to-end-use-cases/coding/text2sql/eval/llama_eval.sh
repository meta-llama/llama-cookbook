# Set to "true" to enable debug mode with detailed prints
DEBUG_MODE="false"

eval_path='../data/dev_20240627/dev.json'
db_root_path='../data/dev_20240627/dev_databases/'
ground_truth_path='../data/'

# Llama models on Llama API
# YOUR_API_KEY='YOUR_LLAMA_API_KEY'
# model='Llama-3.3-8B-Instruct'
#model='Llama-3.3-70B-Instruct'
#model='Llama-4-Maverick-17B-128E-Instruct-FP8'
#model='Llama-4-Scout-17B-16E-Instruct-FP8'

# Llama model on Hugging Face Hub https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
# YOUR_API_KEY='huggingface'
# model='meta-llama/Llama-3.1-8B-Instruct'

# Fine-tuned Llama models locally
YOUR_API_KEY='finetuned'
model='../fine-tuning/llama31-8b-text2sql-fft-nonquantized-cot-epochs-3'

data_output_path="./output/$model/"

echo "Text2SQL using $model"
python3 -u llama_text2sql.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} \
--model ${model} --eval_path ${eval_path} --data_output_path ${data_output_path}

# Check if llama_text2sql.py exited successfully
if [ $? -eq 0 ]; then
    echo "llama_text2sql.py completed successfully. Proceeding with evaluation..."

    # Add --debug flag if DEBUG_MODE is true
    if [ "$DEBUG_MODE" = "true" ]; then
        python3 -u text2sql_eval.py --db_root_path ${db_root_path} --predicted_sql_path ${data_output_path} \
        --ground_truth_path ${ground_truth_path} \
        --diff_json_path ${eval_path} --debug
    else
        python3 -u text2sql_eval.py --db_root_path ${db_root_path} --predicted_sql_path ${data_output_path} \
        --ground_truth_path ${ground_truth_path} \
        --diff_json_path ${eval_path}
    fi

    echo "Done evaluating $model."

else
    echo "Error: llama_text2sql.py failed with exit code $?. Skipping evaluation."
    exit 1
fi
