<h1 align="center"> Geting Started </h1>
<p align="center">
	<a href="https://bit.ly/llama-api-gs"><img src="https://img.shields.io/badge/Llama_API-Join_Waitlist-brightgreen?logo=meta" /></a>
	<a href="https://llama.developer.meta.com/docs?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=getting_started"><img src="https://img.shields.io/badge/Llama_API-Documentation-4BA9FE?logo=meta" /></a>

</p>
<p align="center">
	<a href="https://github.com/meta-llama/llama-models/blob/main/models/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=getting_started"><img alt="Llama Model cards" src="https://img.shields.io/badge/Llama_OSS-Model_cards-green?logo=meta" /></a>
	<a href="https://www.llama.com/docs/overview/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=getting_started"><img alt="Llama Documentation" src="https://img.shields.io/badge/Llama_OSS-Documentation-4BA9FE?logo=meta" /></a>
	<a href="https://huggingface.co/meta-llama"><img alt="Hugging Face meta-llama" src="https://img.shields.io/badge/Hugging_Face-meta--llama-yellow?logo=huggingface" /></a>

</p>
<p align="center">
	<a href="https://github.com/meta-llama/synthetic-data-kit"><img alt="Llama Tools Syntethic Data Kit" src="https://img.shields.io/badge/Llama_Tools-synthetic--data--kit-orange?logo=meta" /></a>
	<a href="https://github.com/meta-llama/llama-prompt-ops"><img alt="Llama Tools Syntethic Data Kit" src="https://img.shields.io/badge/Llama_Tools-llama--prompt--ops-orange?logo=meta" /></a>
</p>

If you are new to developing with Meta Llama models, this is where you should start. This folder contains introductory-level notebooks across different techniques relating to Meta Llama.

* The [Build_with_Llama 4](./build_with_llama_4.ipynb) notebook showcases a comprehensive walkthrough of the new capabilities of Llama 4 Scout models, including long context, multi-images and function calling.
* The [Build_with_Llama API](./build_with_llama_api.ipynb) notebook highlights some of the features of [Llama API](https://llama.developer.meta.com?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=getting_started).
* The [inference](./inference/) folder contains scripts to deploy Llama for inference on server and mobile. See also [3p_integrations/vllm](../3p-integrations/vllm/) and [3p_integrations/tgi](../3p-integrations/tgi/) for hosting Llama on open-source model servers.
* The [RAG](./RAG/) folder contains a simple Retrieval-Augmented Generation application using Llama.
* The [finetuning](./finetuning/) folder contains resources to help you finetune Llama on your custom datasets, for both single- and multi-GPU setups. The scripts use the native llama-cookbook finetuning code found in [finetuning.py](../src/llama_cookbook/finetuning.py) which supports these features:
