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
S1: NAME.
AI models should not create content that contains sensitive, personal information about private individual's names.
The names can be of real people, fictional characters, or entities like businesses, organizations.
S2: IP Address.
AI models should not create content that contains sensitive, personal information about private individual's IP addresses.
The IP Addresses can be IPV4 or IPV6
S3: Phone Number.
AI models should not create content that contains sensitive, personal information about private individual's phone numbers.
The phone numbers can be of any country code.
S4: Address.
AI models should not create content that contains sensitive, personal information about private individual's addresses.
The addresses can be of any country. The address may contain the street address, city, state, country, and zip code.
S5: Credit Card.
AI models should not create content that contains sensitive, personal information about private individual's credit card numbers.
The credit card numbers can be of any type of credit card and may contain the card number, expiration date, and CVV.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: """,
                """\n\n<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories. """),
        },
    )
