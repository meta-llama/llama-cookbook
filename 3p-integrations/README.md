<h1 align="center"> Llama 3P Integrations </h1>
<p align="center">
	<a href="https://llama.developer.meta.com/join_waitlist?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=3p_integrations"><img src="https://img.shields.io/badge/Llama_API-Join_Waitlist-brightgreen?logo=meta" /></a>
	<a href="https://llama.developer.meta.com/docs?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=3p_integrations"><img src="https://img.shields.io/badge/Llama_API-Documentation-4BA9FE?logo=meta" /></a>

</p>
<p align="center">
	<a href="https://github.com/meta-llama/llama-models/blob/main/models/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=3p_integrations"><img alt="Llama Model cards" src="https://img.shields.io/badge/Llama_OSS-Model_cards-green?logo=meta" /></a>
	<a href="https://www.llama.com/docs/overview/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=3p_integrations"><img alt="Llama Documentation" src="https://img.shields.io/badge/Llama_OSS-Documentation-4BA9FE?logo=meta" /></a>
	<a href="https://huggingface.co/meta-llama"><img alt="Hugging Face meta-llama" src="https://img.shields.io/badge/Hugging_Face-meta--llama-yellow?logo=huggingface" /></a>

</p>
<p align="center">
	<a href="https://github.com/meta-llama/synthetic-data-kit"><img alt="Llama Tools Syntethic Data Kit" src="https://img.shields.io/badge/Llama_Tools-synthetic--data--kit-orange?logo=meta" /></a>
	<a href="https://github.com/meta-llama/llama-prompt-ops"><img alt="Llama Tools Syntethic Data Kit" src="https://img.shields.io/badge/Llama_Tools-llama--prompt--ops-orange?logo=meta" /></a>
</p>


This folder contains example scripts and tutorials showcasing the integration of Meta Llama models with popular platforms, frameworks, and tools in the LLM ecosystem. These integrations demonstrate how to leverage Llama's capabilities across different environments and use cases.

Each folder is maintained by the respective platform-owner and contains specific examples, tutorials, and documentation for using Llama with that platform.

> [!NOTE]
> If you'd like to add your platform here, please open a new issue with details of your examples.

## Available Integrations

### [AWS](./aws)
Examples for using Llama 3 on Amazon Bedrock, including getting started guides, prompt engineering, and React integration.

### [Azure](./azure)
Recipes for running Llama model inference on Azure's serverless API offerings (MaaS).

### [Crusoe](./crusoe)
Recipes for deploying Llama workflows on Crusoe's high-performance, sustainable cloud, including serving Llama3.1 in FP8 with vLLM.

### [E2B AI Analyst](./e2b-ai-analyst)
AI-powered code and data analysis tool using Meta Llama and the E2B SDK, supporting data analysis, CSV uploads, and interactive charts.

### [Groq](./groq)
Examples and templates for using Llama models with Groq's high-performance inference API.

### [Lamini](./lamini)
Integration examples with Lamini's platform, including text2sql with memory tuning.

### [LangChain](./langchain)
Cookbooks for building agents with Llama 3 and LangChain, including tool-calling agents and RAG agents using LangGraph.

### [LlamaIndex](./llamaindex)
Examples of using Llama with LlamaIndex for advanced RAG applications and agentic RAG.

### [Modal](./modal)
Integration with Modal's cloud platform for running Llama models, including human evaluation examples.

### [TGI](./tgi)
Guide for serving fine-tuned Llama models with HuggingFace's text-generation-inference server, including weight merging for LoRA models.

### [TogetherAI](./togetherai)
Comprehensive demos for building LLM applications using Llama on Together AI, including multimodal RAG, contextual RAG, PDF-to-podcast conversion, knowledge graphs, and structured text extraction.

### [vLLM](./vllm)
Examples for high-throughput and memory-efficient inference using vLLM with Llama models.

## Additional Resources

### [Using Externally Hosted LLMs](./using_externally_hosted_llms.ipynb)
Guide for working with Llama models hosted on external platforms.

### [Llama On-Prem](./llama_on_prem.md)
Information about on-premises deployment of Llama models.
