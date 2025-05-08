# LangChain + Llama 3 Cookbooks

## Understanding Workflows and Agents

Agentic systems can be implemented as either "workflows" or "agents":

* **Workflows**: Systems where LLMs and tools are orchestrated through predefined code paths
* **Agents**: Systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks

Learn more about this distinction in the [LangGraph documentation](https://langchain-ai.github.io/langgraph/tutorials/workflows/).

[LangGraph](https://langchain-ai.github.io/langgraph/concepts/high_level/) is a powerful library for building both workflows and agents, offering benefits such as:
- Persistence
- Streaming
- Debugging support
- Deployment capabilities

These notebooks demonstrate how to build effective agents and workflows with LangGraph using Llama models.

![LangGraph Overview](https://github.com/rlancemartin/llama-recipes/assets/122662504/a2c2ec40-2c7b-486e-9290-33b6da26c304)

## Notebooks in this Collection

### 1. LangGraph Tool Calling Agent

In [`langgraph_tool_calling_agent.ipynb`](./langgraph_tool_calling_agent.ipynb), we demonstrate how to build an agent using LangGraph with tool calling capabilities. This implementation shows how to create flexible, dynamic agents that can select and use tools to complete tasks.

Watch the [video overview](https://www.youtube.com/watch?v=j2OAeeujQ9M) for a detailed explanation of this agent's design.

### 2. LangGraph RAG Workflow

In [`langgraph_rag_workflow.ipynb`](./langgraph_rag_workflow.ipynb), we show how to build a custom Llama-powered RAG workflow that incorporates ideas from three research papers:

* **Corrective-RAG (CRAG)** [paper](https://arxiv.org/pdf/2401.15884.pdf): Uses self-grading on retrieved documents and web-search fallback when documents aren't relevant
* **Self-RAG** [paper](https://arxiv.org/abs/2310.11511): Adds self-grading on generations to detect hallucinations and evaluate answer quality
* **Adaptive RAG** [paper](https://arxiv.org/abs/2403.14403): Routes queries between different RAG approaches based on query complexity

We implement these approaches as control flows in LangGraph with three key components:

- **Planning:** The sequence of RAG steps (retrieval, grading, generation)
- **Memory:** RAG-related information (questions, retrieved documents) passed between steps
- **Tool use:** Tools for RAG operations (deciding between web search or vectorstore retrieval)

The workflow progressively builds from CRAG (blue) to Self-RAG (green) to Adaptive RAG (red):

![RAG Workflow Evolution](https://github.com/rlancemartin/llama-recipes/assets/122662504/ec4aa1cd-3c7e-4cd1-a1e7-7deddc4033a8)

This implementation demonstrates how workflows can constrain control flow, enabling effective operation even with low-capacity local LLMs.

Watch the [video overview](https://www.youtube.com/watch?v=sgnrL7yo1TE) for a detailed explanation of this workflow's design.
