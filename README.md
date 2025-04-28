<h1 align="center"> Llama Cookbook </h1>
<p align="center">
	<a href="https://bit.ly/llama-api-main"><img src="https://img.shields.io/badge/Llama_API-Join_Waitlist-brightgreen?logo=meta" /></a>
	<a href="https://llama.developer.meta.com/docs?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img src="https://img.shields.io/badge/Llama_API-Documentation-4BA9FE?logo=meta" /></a>

</p>
<p align="center">
	<a href="https://github.com/meta-llama/llama-models/blob/main/models/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img alt="Llama Model cards" src="https://img.shields.io/badge/Llama_OSS-Model_cards-green?logo=meta" /></a>
	<a href="https://www.llama.com/docs/overview/?utm_source=llama-cookbook&utm_medium=readme&utm_campaign=main"><img alt="Llama Documentation" src="https://img.shields.io/badge/Llama_OSS-Documentation-4BA9FE?logo=meta" /></a>
	<a href="https://huggingface.co/meta-llama"><img alt="Hugging Face meta-llama" src="https://img.shields.io/badge/Hugging_Face-meta--llama-yellow?logo=huggingface" /></a>

</p>
<p align="center">
	<a href="https://github.com/meta-llama/synthetic-data-kit"><img alt="Llama Tools Syntethic Data Kit" src="https://img.shields.io/badge/Llama_Tools-synthetic--data--kit-orange?logo=meta" /></a>
	<a href="https://github.com/meta-llama/llama-prompt-ops"><img alt="Llama Tools Syntethic Data Kit" src="https://img.shields.io/badge/Llama_Tools-llama--prompt--ops-orange?logo=meta" /></a>
</p>
<h2> Official Guide to building with Llama </h2>

Checkout our latest model tutorial here: [Build with Llama 4 Scout](./getting-started/build_with_llama_4.ipynb)

Welcome to the official repository for helping you get started with [inference](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/inference/), [fine-tuning](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/finetuning) and [end-to-end use-cases](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases) of building with the Llama Model family.

This repository covers the most popular community approaches, use-cases and the latest recipes for Llama Text and Vision models.

> [!TIP]
> Popular getting started links:
> * [Build with Llama 4 Scout](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/build_with_llama_4.ipynb)
> * [Multimodal Inference with Llama 3.2 Vision](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/inference/local_inference/README.md#multimodal-inference)
> * [Inferencing using Llama Guard (Safety Model)](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/responsible_ai/llama_guard/)

> [!TIP]
> Popular end to end recipes:
> * [Email Agent](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/email_agent/)
> * [NotebookLlama](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/NotebookLlama/)
> * [Text to SQL](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases/coding/text2sql/)


> Note: We recently did a refactor of the repo, [archive-main](https://github.com/meta-llama/llama-cookbook/tree/archive-main) is a snapshot branch from before the refactor

## Repository Structure:

- [3P Integrations](https://github.com/meta-llama/llama-cookbook/tree/main/3p-integrations): Getting Started Recipes and End to End Use-Cases from various Llama providers
- [End to End Use Cases](https://github.com/meta-llama/llama-cookbook/tree/main/end-to-end-use-cases): As the name suggests, spanning various domains and applications
- [Getting Started](https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/): Reference for inferencing, fine-tuning and RAG examples
- [src](https://github.com/meta-llama/llama-cookbook/tree/main/src/): Contains the src for the original llama-recipes library along with some FAQs for fine-tuning.

## FAQ:

- **Q:** What happened to llama-recipes?
  **A:** We recently renamed llama-recipes to llama-cookbook.

- **Q:** Prompt Template changes for Multi-Modality?
  **A:** Llama 3.2 follows the same prompt template as Llama 3.1, with a new special token `<|image|>` representing the input image for the multimodal models. More details on the prompt templates for image reasoning, tool-calling, and code interpreter can be found [on the documentation website](https://www.llama.com/docs/overview).

- **Q:** I have some questions for Fine-Tuning, is there a section to address these?
  **A:** Check out the Fine-Tuning FAQ [here](https://github.com/meta-llama/llama-cookbook/tree/main/src/docs/).

- **Q:** Some links are broken/folders are missing:
  **A:** We recently did a refactor of the repo, [archive-main](https://github.com/meta-llama/llama-cookbook/tree/archive-main) is a snapshot branch from before the refactor.

- **Q:** Where can we find details about the latest models?
  **A:** Official [Llama models website](https://www.llama.com).

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License
<!-- markdown-link-check-disable -->
See the License file for Meta Llama 4 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama4/USE_POLICY.md)

See the License file for Meta Llama 3.3 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_3/USE_POLICY.md)

See the License file for Meta Llama 3.2 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/USE_POLICY.md)

See the License file for Meta Llama 3.1 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/USE_POLICY.md)

See the License file for Meta Llama 3 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3/USE_POLICY.md)

See the License file for Meta Llama 2 [here](https://github.com/meta-llama/llama-models/blob/main/models/llama2/LICENSE) and Acceptable Use Policy [here](https://github.com/meta-llama/llama-models/blob/main/models/llama2/USE_POLICY.md)
<!-- markdown-link-check-enable -->
