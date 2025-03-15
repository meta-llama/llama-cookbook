import gradio as gr

from prompt_utils import ChatFormat, Dialog

SINGLE_TURN = "Single Turn"
MULTI_TURN = "Multi Turn"

def prompt_template_dropdown_listener(value):
    # This function will be called whenever the value of the dropdown changes
    if value == SINGLE_TURN:
        return {
            assistant_response: gr.Textbox(elem_id="assistant_response", visible = False),
            user_prompt_2: gr.Textbox(elem_id="user_prompt_2", visible = False)
        }
    else:
        return {
            assistant_response: gr.Textbox(elem_id="assistant_response", visible = True),
            user_prompt_2: gr.Textbox(elem_id="user_prompt_2", visible = True)
        }

def format_prompt_template_listener(system_prompt, user_prompt_1, assistant_response, user_prompt_2, prompt_template):
    if not user_prompt_1:
        raise gr.Error("User prompt is mandatory.")

    if not user_prompt_2 and assistant_response:
        raise gr.Error("When the assistant message is set, the second user prompt is mandatory.")

    if prompt_template == MULTI_TURN and ((user_prompt_2 and not assistant_response) or (not user_prompt_2 and not assistant_response)):
        raise gr.Error("When generating a multi turn prompt, the assistant message is mandatory.")

    dialog: Dialog = []
    if system_prompt: dialog.append({   "role": "system", "content": system_prompt, })
    dialog.append({   "role": "user", "content": user_prompt_1, })
    if assistant_response: dialog.append({   "role": "assistant", "content": assistant_response, })
    if user_prompt_2: dialog.append({   "role": "user", "content": user_prompt_2, })

    return {
        prompt_output: ChatFormat.encode_dialog_prompt(dialog),
        python_output: ChatFormat.format_python_prompt_output(system_prompt, user_prompt_1, assistant_response, user_prompt_2),
        hf_output: "",
    }


with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            gr.Markdown("## Configurations")
            model = gr.Dropdown(["Llama 3", "Llama 2"], label="Model", filterable=False)

            prompt_template = gr.Dropdown([SINGLE_TURN, MULTI_TURN], label="Prompt Template", filterable=False, value=SINGLE_TURN)

            gr.Markdown("## Input Prompts")
            system_prompt = gr.Textbox(label="System prompt", lines=2, placeholder="Optional System prompt for the model")
            user_prompt_1 = gr.Textbox(elem_id="user_prompt_1", label="User prompt *", lines=2, placeholder="User prompt required")
            assistant_response = gr.Textbox(label="Assistant response", lines=2, visible=False, elem_id="assistant_response", placeholder="Assistant prompt required on Multi Turn")
            user_prompt_2 = gr.Textbox(label="User prompt *", lines=2, visible=False, elem_id="user_prompt_2", placeholder="User prompt required on Multi turn")

            prompt_template.input(prompt_template_dropdown_listener, prompt_template, [assistant_response, user_prompt_2])

            submit = gr.Button("Submit")

        with gr.Column(scale=3, min_width=600):
            gr.Markdown("## Output")
            with gr.Tab("Preview"):
                prompt_output = gr.Code(show_label=True, interactive=False, min_width=600, lines=30)

            with gr.Tab("Code"):
                with gr.Row():
                    with gr.Tab("Plain Python"):
                        python_output = gr.Code(label="Python Code", interactive=False, min_width=600, lines=30, language="python")
                    with gr.Tab("Hugging Face"):
                        hf_output = gr.Textbox(label="Using HF Transformers", interactive=False, min_width=600, lines=30, show_copy_button=True)

    inputs = [system_prompt, user_prompt_1, assistant_response, user_prompt_2, prompt_template]
    outputs = [prompt_output, python_output, hf_output]
    submit.click(format_prompt_template_listener, inputs, outputs)

demo.launch()
