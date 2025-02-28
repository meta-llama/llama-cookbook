# Contextual keywords generation for RAG using Llama-3.1

**Problem**: Independent chunking in traditional RAG systems leads to the loss of contextual information between chunks. This makes it difficult for LLMs to retrieve relevant data when context (e.g., the subject or entity being discussed) is not explicitly repeated within individual chunks.

**Solution**: Generate keywords for each chunk to fulfill missing contextual information. These keywords (e.g., "BMW, X5, pricing") enrich the chunk with necessary context, ensuring better retrieval accuracy. By embedding this enriched metadata, the system bridges gaps between related chunks, enabling effective query matching and accurate answer generation.

[This article](https://medium.com/@ailabs/overcoming-independent-chunking-in-rag-systems-a-hybrid-approach-5d2c205b3732) explains benefits of contextual chunking.

**Note** This method does not require calling LLM for each chunk separately, which makes it efficient.

**Getting started**
In this cookbook, we’ll use DeepInfra for Llama inference services, so be sure to obtain an API key from https://deepinfra.com/.
You'll also need a LlamaParse API key to parse PDF files, which can be obtained from https://www.llamaindex.ai/.
Additionally, an OpenAI API key is required for embeddings.
Before getting started, update the <code>config.py</code> file as following:
    "DEEPINFRA_API_KEY"="<your_api_key>"
    "OPENAI_API_KEY"="<your_api_key>"
    "LLAMAPARSE_API_KEY"="<your_api_key>"