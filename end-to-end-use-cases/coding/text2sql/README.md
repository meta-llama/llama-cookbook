## Text2SQL: Eval and Fine-tuning Tools and Quick Start Notebook

This folder contains the `tool` subfolder, which has e2e scripts for evaluating Llama (original and fine-tuned) models on the Text2SQL task using the popular [BIRD](https://bird-bench.github.io) dataset, and e2e scripts for generating fine-tuning datasets and fine-tuning Llama 3.1 8B with the datasets.

Before looking into the `tool` folder, you may start with the scripts and notebook in this folder to get familiar with how to interact with a database using natural language inputs bu asking Llama to convert natural language queries into SQL queries.

For detailed instructions on setting up the environment, creating a database, and executing natural language queries using the Text2SQL interface, please refer to the [quickstart.ipynb](quickstart.ipynb) notebook.

### Structure:

- tool: A folder containing scripts for evaluating and fine-tuning Llama models on the Text2SQL task.
- quickstart.ipynb: A Quick Demo of Text2SQL Using Llama 3.3. This Jupyter Notebook includes examples of how to use the interface to execute natural language queries on the sample data. It uses Llama 3.3 to answer questions about a SQLite database using LangChain and the Llama cloud provider Together.ai.
- nba.txt: A text file containing NBA roster information, which is used as sample data for demonstration purposes.
- txt2csv.py: A script that converts text data into a CSV format. This script is used to preprocess the input data before it is fed into csv2db.py.
- csv2db.py: A script that imports data from a CSV file into a SQLite database. This script is used to populate the database with sample data.
- nba_roster.db: A SQLite database file created from the nba.txt data, used to test the Text2SQL interface.
