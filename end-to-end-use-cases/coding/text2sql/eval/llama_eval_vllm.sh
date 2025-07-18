eval_path='../data/dev_20240627/dev.json'
db_root_path='../data/dev_20240627/dev_databases/'
ground_truth_path='../data/'

model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
data_output_path="./output/$model/"

echo "Text2SQL using $model"
python3 -u llama_text2sql_vllm.py --db_root_path ${db_root_path} \
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
