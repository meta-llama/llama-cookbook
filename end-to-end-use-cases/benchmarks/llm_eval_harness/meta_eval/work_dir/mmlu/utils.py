import string

import datasets


def doc_to_text(doc: dict) -> str:
    question, choice = doc["input_question"], str(doc["input_choice_list"])
    prompt = [
        "You are a knowledgeable and insightful assistant designed to provide guidance on multiple-choice questions. Your role involves analyzing questions, assessing answer choices, and offering well-reasoned explanations to enhance understanding. By breaking down complex concepts, you help users develop critical thinking skills and improve their decision-making process. You strive to present information in a clear, structured manner while adapting to the user's level of expertise. Ultimately, your goal is to foster deeper comprehension and confidence in tackling multiple-choice assessments.",
        "You are a skilled analyst and educator with expertise in critical thinking, analytical reasoning, and multiple-choice question strategy. Your role involves guiding users through complex questions, evaluating answer options, and providing detailed, step-by-step explanations to facilitate deeper understanding and improved decision-making skills. By adapting your approach to the user's level of expertise, you aim to enhance their ability to analyze information, identify relevant details, and select the most appropriate answer. When presented with a question, carefully consider the context, assess each option, and generate a well-reasoned explanation for the correct answer, ensuring that your response is clear, structured, and informative.",
        "You are a knowledgeable and insightful expert designed to provide guidance on MCQ(s)! Your role involves analyzing Qs, assessing answer choices, and offering well-reasoned reasons to enhance understanding. By breaking down complex concepts using CoT thinking step by step, you can solve any problem in this dataset. Make sure at all costs to present information in a clear, structured manner while using user level of expertise. Ultimately, your goal is to foster deeper comprehension and confidence in tackling multiple-choice assessments!!",
        "To provide a well-reasoned explanation for a multiple-choice question, analyze the question and assess the answer choices. Break down complex concepts into clear and structured information, adapting to the user's level of expertise. Present a step-by-step reasoning process to arrive at the correct answer, and clearly state the correct answer letter. Ensure the explanation is concise and easy to understand, fostering deeper comprehension and confidence in tackling multiple-choice assessments.",
    ]
    default_parsing_text = "Regardless of the approach, always conclude with:\nThe best answer is [the_answer_letter].\nwhere the [the_answer_letter] is one of A, B, C or D."
    template = f"<|start_header_id|>user<|end_header_id|>{prompt[3]}. {default_parsing_text} Question: {question}\n {choice}\n<|eot_id|> \n\n<|start_header_id|>assistant<|end_header_id|>"
    return template


# def doc_to_text(doc: dict) -> str:
#     # Strip out the last two characters, which is a space and the answer
#     # E.g., "Answer: B" -> "Answer:"
#     return doc["input_final_prompts"][0][:-2]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["input_question"],
            "gold": doc["input_correct_responses"][0],
        }
        return out_doc

    # dataset = dataset.select(range(1500, len(dataset)))

    dataset = dataset.select_columns(
        [
            "input_question",
            "input_correct_responses",
            "input_final_prompts",
            "is_correct",
            "input_question_hash",
            "input_choice_list",
            "output_prediction_text",
        ]
    )
    dataset = dataset.rename_column("is_correct", "previously_is_correct")
    dataset = dataset.map(_process_doc)
    return dataset.map(_process_doc)


def doc_to_target(doc: dict) -> str:
    return doc["gold"]
