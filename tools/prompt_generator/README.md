# Overview
This Python code is designed to help developers format and encode conversational prompts for use with AI models. It provides a user-friendly interface to input system prompts, user prompts, and assistant responses, and then generates the corresponding Python code to create these prompts programmatically.
# Key Functions
## String prompt
`encode_dialog_prompt(Dialog: dialog) -> str`
This function takes a Dialog object as input, which is a sequence of Message objects. It returns a string that represents the encoded dialog prompt. It uses the encode_message function to encode each message in the dialog, and then joins them together.
## Python prompt generation
`format_python_prompt_output(system_prompt, user_prompt_1, assistant_response, user_prompt_2) -> str`
This function takes four strings as input: a system prompt, a user prompt, an assistant response, and a second user prompt. It returns a string that represents the Python code to create these prompts programmatically. This function is used to generate the code in the Python code tab.
# Usage
To use this code, developers will interact with the Gradio interface to input their prompts and select their options. After clicking the "Submit" button, the Python code to create their prompts will be displayed in the "Python Code" tab. Developers can then copy this code and use it in their own applications.
Please note that the user prompt is mandatory, and if the assistant message is set, the second user prompt is also mandatory. When generating a multi-turn prompt, the assistant message is mandatory.
# How to Run
To run the notebook, you can open it in <a href="https://colab.research.google.com/github/varunfb/llama-recipes/blob/main/tools/prompt_generator/prompt_sample.ipynb"><img data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab" src="https://camo.githubusercontent.com/f5e0d0538a9c2972b5d413e0ace04cecd8efd828d133133933dfffec282a4e1b/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>

To run locally, in this directory, run:
1. `pip install -r requirements.txt`
2. `gradio gradio_ui.py` 
