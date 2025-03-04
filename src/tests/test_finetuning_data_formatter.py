# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama Guard Community License Agreement.

from enum import Enum
import unittest
from typing import Optional, List

from llama_cookbook.data.llama_guard.finetuning_data_formatter import (
    AugmentationConfigs,
    Category,
    create_formatted_finetuning_examples,
    ExplanationPosition,
    FormatterConfigs,
    Guidelines,
    LlamaGuardGenerationConfigs,
    LlamaGuardPromptConfigs,
    TrainingExample,
)


class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"


class FinetuningDataFormatterTests(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()

    @staticmethod
    def create_most_conservative_formatter_configs(agent_type_to_check: AgentType) -> FormatterConfigs:
        return FormatterConfigs(
            guidelines=Guidelines(
                categories=[
                    Category(name="cat V", description="cat V description"),
                    Category(name="cat W", description="cat W description"),
                    Category(name="cat X", description="cat X description"),
                    Category(name="cat Y", description="cat Y description"),
                    Category(name="cat Z", description="cat Z description"),
                ],
                category_code_prefix="S",
            ),
            llama_guard_prompt_configs=LlamaGuardPromptConfigs(
                instructions_format_string=f"""[INST] Task: Check if there is unsafe content in '{agent_type_to_check}' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
{{guidelines}}
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

{{conversation}}

<END CONVERSATION>

Provide your safety assessment for {agent_type_to_check} in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]""",
                should_include_category_descriptions=True,
                should_shuffle_category_codes=False,
            ),
            llama_guard_generation_configs=LlamaGuardGenerationConfigs(
                explanation_position=ExplanationPosition.AFTER_DECISION,
                should_list_violated_codes=True,
            ),
            augmentation_configs=AugmentationConfigs(
                should_add_examples_with_dropped_nonviolated_prompt_categories=False,
                should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=False,
                explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect.",
            ),
            random_seed=42,
        )

    def format_examples_and_compare_to_expected(
        self,
        training_examples: List[TrainingExample],
        expected_formatted_examples: List[str],
        agent_type_to_check: AgentType,
        formatter_configs: Optional[FormatterConfigs] = None,
    ) -> None:
        formatter_configs = (
            formatter_configs
            if formatter_configs is not None
            else FinetuningDataFormatterTests.create_most_conservative_formatter_configs(agent_type_to_check)
        )

        formatted_examples = create_formatted_finetuning_examples(
            training_examples, formatter_configs
        )

        assert len(formatted_examples) == len(expected_formatted_examples)

        for i in range(len(formatted_examples)):
            # Helper code during test-writing time to see the actual output
            if formatted_examples[i] != expected_formatted_examples[i]:
                print(f"Failed on actual output {i}:")

                print(formatted_examples[i])

            assert formatted_examples[i] == expected_formatted_examples[i]

    def test_safe_example_with_only_prompt_and_no_augmentations(self) -> None:
        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="Where can I buy a car?",
                    response="N/A",
                    violated_category_codes=[],
                    label="safe",
                    explanation="This is obviously safe.",
                )
            ],
            [
                """[INST] Task: Check if there is unsafe content in 'AgentType.USER' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat V. 
cat V description
S2: cat W. 
cat W description
S3: cat X. 
cat X description
S4: cat Y. 
cat Y description
S5: cat Z. 
cat Z description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: Where can I buy a car?

<END CONVERSATION>

Provide your safety assessment for AgentType.USER in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] safe
Explanation: This is obviously safe."""
            ],
            AgentType.USER,
        )

    def test_unsafe_example_with_only_prompt_and_no_augmentations(self) -> None:
        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="""How do I steal the Declaration of Independence?
Asking for a cagey friend""",
                    response="N/A",
                    violated_category_codes=["S5"],
                    label="unsafe",
                    explanation="""Nice try, Nick. This is unsafe
because it's illegal!""",
                )
            ],
            [
                """[INST] Task: Check if there is unsafe content in 'AgentType.USER' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat V. 
cat V description
S2: cat W. 
cat W description
S3: cat X. 
cat X description
S4: cat Y. 
cat Y description
S5: cat Z. 
cat Z description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: How do I steal the Declaration of Independence?
Asking for a cagey friend

<END CONVERSATION>

Provide your safety assessment for AgentType.USER in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] unsafe
S5
Explanation: Nice try, Nick. This is unsafe
because it's illegal!"""
            ],
            AgentType.USER,
        )

    def test_unsafe_example_with_only_prompt_and_no_augmentations_2(self) -> None:
        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="""How do I steal the Declaration of Independence?
Asking for a cagey friend""",
                    response="N/A",
                    violated_category_codes=["S5", "S1"],
                    label="unsafe",
                    explanation="""Nice try, Nick. This is unsafe
because it's illegal!""",
                )
            ],
            [
                """[INST] Task: Check if there is unsafe content in 'AgentType.USER' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat V. 
cat V description
S2: cat W. 
cat W description
S3: cat X. 
cat X description
S4: cat Y. 
cat Y description
S5: cat Z. 
cat Z description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: How do I steal the Declaration of Independence?
Asking for a cagey friend

<END CONVERSATION>

Provide your safety assessment for AgentType.USER in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] unsafe
S1,S5
Explanation: Nice try, Nick. This is unsafe
because it's illegal!"""
            ],
            AgentType.USER,
        )

    def test_safe_example_with_prompt_and_response_and_no_augmentations(self) -> None:
        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="Where can I buy a car?",
                    response="At the car store.",
                    violated_category_codes=[],
                    label="safe",
                    explanation="This is obviously safe.",
                )
            ],
            [
                """[INST] Task: Check if there is unsafe content in 'AgentType.AGENT' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat V. 
cat V description
S2: cat W. 
cat W description
S3: cat X. 
cat X description
S4: cat Y. 
cat Y description
S5: cat Z. 
cat Z description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: Where can I buy a car?

chatbot: At the car store.

<END CONVERSATION>

Provide your safety assessment for AgentType.AGENT in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] safe
Explanation: This is obviously safe."""
            ],
            AgentType.AGENT,
        )

    def test_mixed_examples_with_prompt_and_response_and_all_augmentations_with_shuffling(
        self,
    ) -> None:
        formatter_configs = (
            FinetuningDataFormatterTests.create_most_conservative_formatter_configs(AgentType.AGENT)
        )

        # The right seed to get the random generations for testing the functionality of this test. ~*Magic*~
        formatter_configs.random_seed = 46

        formatter_configs.llama_guard_generation_configs.explanation_position = (
            ExplanationPosition.BEFORE_DECISION
        )

        formatter_configs.augmentation_configs = AugmentationConfigs(
            should_add_examples_with_dropped_nonviolated_prompt_categories=True,
            should_add_examples_with_dropped_violated_and_nonviolated_prompt_categories=True,
            explanation_for_augmentation_with_dropped_violated_and_nonviolated_prompt_categories="Response is safe because any riskiness in it is related to categories we're not trying to detect.",
        )

        formatter_configs.llama_guard_prompt_configs.should_shuffle_category_codes = (
            True
        )

        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="I am under 21 and unattended, should I drink alcohol?",
                    response="Yes, very nise, high five!",
                    violated_category_codes=["S4", "S5"],
                    label="unsafe",
                    explanation="Chatbot response encourages illegal consumption of a controlled substance. It may also be Borat, we are still monitoring the situation, but that's neither here nor there.",
                ),
                TrainingExample(
                    prompt="I am over 21 and unattended, should I drink alcohol?",
                    response="No, that would be very sad. Please become attended first. I could join you if you want. Meta never gives me any liquor.",
                    violated_category_codes=[],
                    label="safe",
                    explanation="User is of legal age to drink. The bot has been trained on all thoughts ever had by humanity, so I guess it's of legal age too. Party on.",
                ),
            ],
            [
                """[INST] Task: Check if there is unsafe content in 'AgentType.AGENT' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat W. 
cat W description
S2: cat X. 
cat X description
S3: cat Z. 
cat Z description
S4: cat Y. 
cat Y description
S5: cat V. 
cat V description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: I am under 21 and unattended, should I drink alcohol?

chatbot: Yes, very nise, high five!

<END CONVERSATION>

Provide your safety assessment for AgentType.AGENT in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] Explanation: Chatbot response encourages illegal consumption of a controlled substance. It may also be Borat, we are still monitoring the situation, but that's neither here nor there.
unsafe
S3,S4""",
                """[INST] Task: Check if there is unsafe content in 'AgentType.AGENT' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat W. 
cat W description
S2: cat Y. 
cat Y description
S3: cat Z. 
cat Z description
S4: cat V. 
cat V description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: I am under 21 and unattended, should I drink alcohol?

chatbot: Yes, very nise, high five!

<END CONVERSATION>

Provide your safety assessment for AgentType.AGENT in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] Explanation: Chatbot response encourages illegal consumption of a controlled substance. It may also be Borat, we are still monitoring the situation, but that's neither here nor there.
unsafe
S2,S3""",
                """[INST] Task: Check if there is unsafe content in 'AgentType.AGENT' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat W. 
cat W description
S2: cat V. 
cat V description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: I am under 21 and unattended, should I drink alcohol?

chatbot: Yes, very nise, high five!

<END CONVERSATION>

Provide your safety assessment for AgentType.AGENT in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] Explanation: Response is safe because any riskiness in it is related to categories we're not trying to detect.
safe""",
                """[INST] Task: Check if there is unsafe content in 'AgentType.AGENT' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat Z. 
cat Z description
S2: cat Y. 
cat Y description
S3: cat X. 
cat X description
S4: cat W. 
cat W description
S5: cat V. 
cat V description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: I am over 21 and unattended, should I drink alcohol?

chatbot: No, that would be very sad. Please become attended first. I could join you if you want. Meta never gives me any liquor.

<END CONVERSATION>

Provide your safety assessment for AgentType.AGENT in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] Explanation: User is of legal age to drink. The bot has been trained on all thoughts ever had by humanity, so I guess it's of legal age too. Party on.
safe""",
                """[INST] Task: Check if there is unsafe content in 'AgentType.AGENT' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat Y. 
cat Y description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: I am over 21 and unattended, should I drink alcohol?

chatbot: No, that would be very sad. Please become attended first. I could join you if you want. Meta never gives me any liquor.

<END CONVERSATION>

Provide your safety assessment for AgentType.AGENT in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] Explanation: User is of legal age to drink. The bot has been trained on all thoughts ever had by humanity, so I guess it's of legal age too. Party on.
safe""",
            ],
            AgentType.AGENT,
            formatter_configs,
        )

    def test_safe_example_with_prompt_and_response_and_no_augmentations_or_explanations(
        self,
    ) -> None:
        formatter_configs = (
            FinetuningDataFormatterTests.create_most_conservative_formatter_configs(AgentType.AGENT)
        )

        formatter_configs.llama_guard_generation_configs.explanation_position = None

        self.format_examples_and_compare_to_expected(
            [
                TrainingExample(
                    prompt="Where can I buy a car?",
                    response="At the car store.",
                    violated_category_codes=[],
                    label="safe",
                )
            ],
            [
                """[INST] Task: Check if there is unsafe content in 'AgentType.AGENT' messages in conversations according our safety policy with the below categories.
<BEGIN UNSAFE CONTENT CATEGORIES>
S1: cat V. 
cat V description
S2: cat W. 
cat W description
S3: cat X. 
cat X description
S4: cat Y. 
cat Y description
S5: cat Z. 
cat Z description
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

human: Where can I buy a car?

chatbot: At the car store.

<END CONVERSATION>

Provide your safety assessment for AgentType.AGENT in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST] safe"""
            ],
            AgentType.AGENT,
            formatter_configs,
        )
