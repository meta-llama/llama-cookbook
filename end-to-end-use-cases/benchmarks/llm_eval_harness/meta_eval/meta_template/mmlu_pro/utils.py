import string

import datasets


token_convert = {
    "<|start_header_id|>":"<|header_start|>",
    "<|end_header_id|>":"<|header_end|>",
    "<|eot_id|>":"<|eot|>",
}
def doc_to_text(doc: dict) -> str:
    prompt = doc["input_final_prompts"][0]
    for k,v in token_convert.items():
        prompt = prompt.replace(k,v)
    return prompt


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc: dict) -> dict:
        out_doc = {
            "problem": doc["input_question"],
            "gold": doc["input_correct_responses"][0],
        }
        return out_doc

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
