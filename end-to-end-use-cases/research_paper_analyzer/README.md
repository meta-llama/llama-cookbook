# Research Paper analyzer with Llama4 Maverick

This leverages **Llama 4 Maverick** model to retrieve the references of an arXiv paper and ingest all their content for question-answering without using any RAG to store these information.

## Features

### Leverage Long Context Length
| Model | Meta Llama4 Maverick | Meta Llama4 Scout | OpenAI GPT-4.5 | Claude Sonnet 3.7 |
| ----- | -------------- | -------------- | -------------- | -------------- |
| Context Window | 1M tokens | 10M tokens | 128K tokens | 1K tokens | 200K tokens |

Because of the long context length, the analyzer can process all the reference paper content at once, so you can ask questions about the paper without worrying about the context length.


## Getting Started

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the application:

```
python research-analyzer.py
```

3. Provide a paper url such as https://arxiv.org/abs/2305.11135

4. Press "Ingest", wait for paper to be processed and ask questions about it
