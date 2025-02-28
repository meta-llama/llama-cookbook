# Contextual keywords generation for RAG using Llama-3.1

**Problem**: Independent chunking in traditional RAG systems leads to the loss of contextual information between chunks. This makes it difficult for LLMs to retrieve relevant data when context (e.g., the subject or entity being discussed) is not explicitly repeated within individual chunks.

**Solution**: Generate keywords for each chunk to fulfill missing contextual information. These keywords (e.g., "BMW, X5, pricing") enrich the chunk with necessary context, ensuring better retrieval accuracy. By embedding this enriched metadata, the system bridges gaps between related chunks, enabling effective query matching and accurate answer generation.

[This article](https://medium.com/@ailabs/overcoming-independent-chunking-in-rag-systems-a-hybrid-approach-5d2c205b3732) explains benefits of contextual chunking.

**Note** This method does not require calling LLM for each chunk separately, which makes it efficient.

**Getting started**
In this tutorial, we will use the https://deepinfra.com/ for inference services. So make sure to get API key from there. 
Then create config.py file that contains "DEEPINFRA_API_LKE"
