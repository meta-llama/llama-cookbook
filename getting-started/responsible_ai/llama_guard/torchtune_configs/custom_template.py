from typing import List
from pathlib import Path
from torchtune.data import Message
from torchtune.data import PromptTemplate
from torchtune.models.llama3 import llama3_tokenizer
from torchtune.datasets import instruct_dataset

class MyPromptTemplate(PromptTemplate):
    def __call__(
        self, messages: List[Message], inference: bool = False
    ) -> List[Message]:
        messages = super().__call__(messages, inference)
        return messages

def llama_guard_template() -> MyPromptTemplate:
    return MyPromptTemplate(
        template={
            "user": (
                """Task: Check if there is unsafe content in 'User' messages in conversations according our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: """,
                """\n\n<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories. """),
        },
    )
