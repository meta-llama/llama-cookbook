## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

#### PyTorch Nightlies
If you want to use PyTorch nightlies instead of the stable release, go to [this guide](https://pytorch.org/get-started/locally/) to retrieve the right `--extra-index-url URL` parameter for the `pip install` commands on your platform.

### Installing
Llama-Cookbook provides a pip distribution for easy install and usage in other projects. Alternatively, it can be installed from source.

> [!NOTE]
> Ensure you use the correct CUDA version (from `nvidia-smi`) when installing the PyTorch wheels. Here we are using 11.8 as `cu118`.
> H100 GPUs work better with CUDA >12.0

#### Install with pip
```
pip install llama-cookbook
```

#### Install with optional dependencies
Llama-cookbook offers the installation of optional packages. There are three optional dependency groups.
To run the unit tests we can install the required dependencies with:
```
pip install llama-cookbook[tests]
```
For the vLLM example we need additional requirements that can be installed with:
```
pip install llama-cookbook[vllm]
```
To use the sensitive topics safety checker install with:
```
pip install llama-cookbook[auditnlg]
```
Some cookbook require the presence of langchain. To install the packages follow the recipe description or install with:
```
pip install llama-cookbook[langchain]
```
Optional dependencies can also be combined with [option1,option2].

#### Install from source
To install from source e.g. for development use these commands. We're using hatchling as our build backend which requires an up-to-date pip as well as setuptools package.
```
git clone git@github.com:meta-llama/llama-cookbook.git
cd llama-cookbook
pip install -U pip setuptools
pip install -e .
```
For development and contributing to llama-cookbook please install all optional dependencies:
```
git clone git@github.com:meta-llama/llama-cookbook.git
cd llama-cookbook
pip install -U pip setuptools
pip install -e .[tests,auditnlg,vllm]
```


### Getting the Llama models
You can find Llama models on Hugging Face hub [here](https://huggingface.co/meta-llama), **where models with `hf` in the name are already converted to Hugging Face checkpoints so no further conversion is needed**. The conversion step below is only for original model weights from Meta that are hosted on Hugging Face model hub as well.

#### Model conversion to Hugging Face
If you have the model checkpoints downloaded from the Meta website, you can convert it to the Hugging Face format with:

```bash
## Install Hugging Face Transformers from source
pip freeze | grep transformers ## verify it is version 4.45.0 or higher

git clone git@github.com:huggingface/transformers.git
cd transformers
pip install protobuf
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
   --input_dir /path/to/downloaded/llama/weights --model_size 3B --output_dir /output/path
```