from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
)

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

class ChatFormat:

    @staticmethod
    def encode_header( message: Message) -> List[str]:
        prompts = []
        prompts.append("<|start_header_id|>")
        prompts.extend(message["role"])
        prompts.append("<|end_header_id|>")
        prompts.extend("\n\n")
        return prompts

    @staticmethod
    def encode_message(message: Message) -> List[str]:
        prompts = ChatFormat.encode_header(message)
        prompts.extend(message["content"].strip())
        prompts.append("<|eot_id|>")
        return prompts

    @staticmethod
    def encode_dialog_prompt(dialog: Dialog) -> str:
        prompts = ["<|begin_of_text|>"]
        for message in dialog:
            prompts.extend(ChatFormat.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        prompts.extend(ChatFormat.encode_header({"role": "assistant", "content": ""}))
        return "".join(prompts)


    @staticmethod
    def format_python_prompt_output(system_prompt, user_prompt_1, assistant_response, user_prompt_2) -> str:
        template = f"""from typing import (
    List,
    Literal,
    Sequence,
    TypedDict,
)

Role = Literal["system", "user", "assistant"]

class Message(TypedDict):
    role: Role
    content: str


Dialog = Sequence[Message]

class ChatFormat:

    @staticmethod
    def encode_header( message: Message) -> List[str]:
        prompts = []
        prompts.append("<|start_header_id|>")
        prompts.extend(message["role"])
        prompts.append("<|end_header_id|>")
        prompts.extend("\\n\\n")
        return prompts

    @staticmethod
    def encode_message(message: Message) -> List[str]:
        prompts = ChatFormat.encode_header(message)
        prompts.extend(message["content"].strip())
        prompts.append("<|eot_id|>")
        return prompts

    @staticmethod
    def encode_dialog_prompt(dialog: Dialog) -> str:
        prompts = ["<|begin_of_text|>"]
        for message in dialog:
            prompts.extend(ChatFormat.encode_message(message))
        # Add the start of an assistant message for the model to complete.
        prompts.extend(ChatFormat.encode_header({{"role": "assistant", "content": ""}}))
        return "".join(prompts)

if __name__ == "__main__":
    dialog: Dialog = []"""
        if system_prompt:
            system_section = f"""
    dialog.append({{   "role": "system", "content": "{system_prompt}", }})"""
            template += system_section
        user_section = f"""
    dialog.append({{   "role": "user", "content": "{user_prompt_1}", }})"""
        template += user_section
        if assistant_response:
            assistant_section = f"""
    dialog.append({{   "role": "assistant", "content": "{assistant_response}", }})"""
            template += assistant_section
        if user_prompt_2:
            second_user_prompt_section = f"""
    dialog.append({{   "role": "user", "content": "{user_prompt_2}", }})"""
            template += second_user_prompt_section

        template += f"""
    print(ChatFormat.encode_dialog_prompt(dialog))"""

        return template
