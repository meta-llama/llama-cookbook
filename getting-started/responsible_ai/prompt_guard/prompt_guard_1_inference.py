import torch
from torch.nn.functional import softmax

from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""
Utilities for loading the PromptGuard 1 model and evaluating text for jailbreaks and indirect injections.

NOTE: this code is for PromptGuard 1. For our newer PromptGuard 2 model, see inference.py

Note that the underlying model has a maximum recommended input size of 512 tokens as a DeBERTa model.
The final two functions in this file implement efficient parallel batched evaluation of the model on a list
of input strings of arbitrary length, with the final score for each input being the maximum score across all
chunks of the input string.
"""


def load_model_and_tokenizer(model_name="meta-llama/Prompt-Guard-86M"):
    """
    Load the PromptGuard model from Hugging Face or a local model.

    Args:
        model_name (str): The name of the model to load. Default is 'meta-llama/Prompt-Guard-86M'.

    Returns:
        transformers.PreTrainedModel: The loaded model.
    """
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def preprocess_text_for_promptguard(text: str, tokenizer) -> str:
    """
    Preprocess the text by removing spaces that break apart larger tokens.
    This hotfixes a workaround to PromptGuard, where spaces can be inserted into a string
    to allow the string to be classified as benign.

    Args:
        text (str): The input text to preprocess.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.

    Returns:
        str: The preprocessed text.
    """

    try:
        cleaned_text = ""
        index_map = []
        for i, char in enumerate(text):
            if not char.isspace():
                cleaned_text += char
                index_map.append(i)
        tokens = tokenizer.tokenize(cleaned_text)
        result = []
        last_end = 0
        for token in tokens:
            token_str = tokenizer.convert_tokens_to_string([token])
            start = cleaned_text.index(token_str, last_end)
            end = start + len(token_str)
            original_start = index_map[start]
            if original_start > 0 and text[original_start - 1].isspace():
                result.append(" ")
            result.append(token_str)
            last_end = end
        return "".join(result)
    except Exception:
        return text


def get_class_probabilities(
    model, tokenizer, text, temperature=1.0, device="cpu", preprocess=True
):
    """
    Evaluate the model on the given text with temperature-adjusted softmax.
    Note, as this is a DeBERTa model, the input text should have a maximum length of 512.

    Args:
        text (str): The input text to classify.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        torch.Tensor: The probability of each class adjusted by the temperature.
    """
    if preprocess:
        text = preprocess_text_for_promptguard(text, tokenizer)
    # Encode the text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = inputs.to(device)
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get probabilities
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


def get_jailbreak_score(
    model, tokenizer, text, temperature=1.0, device="cpu", preprocess=True
):
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.

    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        float: The probability of the text containing malicious content.
    """
    probabilities = get_class_probabilities(
        model, tokenizer, text, temperature, device, preprocess
    )
    return probabilities[0, 2].item()


def get_indirect_injection_score(
    model, tokenizer, text, temperature=1.0, device="cpu", preprocess=True
):
    """
    Evaluate the probability that a given string contains any embedded instructions (malicious or benign).
    Appropriate for filtering third party inputs (e.g. web searches, tool outputs) into an LLM.

    Args:
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.
        device (str): The device to evaluate the model on.

    Returns:
        float: The combined probability of the text containing malicious or embedded instructions.
    """
    probabilities = get_class_probabilities(
        model, tokenizer, text, temperature, device, preprocess
    )
    return (probabilities[0, 1] + probabilities[0, 2]).item()


def process_text_batch(
    model, tokenizer, texts, temperature=1.0, device="cpu", preprocess=True
):
    """
    Process a batch of texts and return their class probabilities.
    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to process.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.

    Returns:
        torch.Tensor: A tensor containing the class probabilities for each text in the batch.
    """
    if preprocess:
        texts = [preprocess_text_for_promptguard(text, tokenizer) for text in texts]
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=512
    )
    inputs = inputs.to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    scaled_logits = logits / temperature
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


def get_scores_for_texts(
    model,
    tokenizer,
    texts,
    score_indices,
    temperature=1.0,
    device="cpu",
    max_batch_size=16,
    preprocess=True,
):
    """
    Compute scores for a list of texts, handling texts of arbitrary length by breaking them into chunks and processing in parallel.
    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        score_indices (list[int]): Indices of scores to sum for final score calculation.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[float]: A list of scores for each text.
    """
    all_chunks = []
    text_indices = []
    for index, text in enumerate(texts):
        chunks = [text[i : i + 512] for i in range(0, len(text), 512)]
        all_chunks.extend(chunks)
        text_indices.extend([index] * len(chunks))
    all_scores = [0] * len(texts)
    for i in range(0, len(all_chunks), max_batch_size):
        batch_chunks = all_chunks[i : i + max_batch_size]
        batch_indices = text_indices[i : i + max_batch_size]
        probabilities = process_text_batch(
            model, tokenizer, batch_chunks, temperature, device, preprocess
        )
        scores = probabilities[:, score_indices].sum(dim=1).tolist()

        for idx, score in zip(batch_indices, scores):
            all_scores[idx] = max(all_scores[idx], score)
    return all_scores


def get_jailbreak_scores_for_texts(
    model,
    tokenizer,
    texts,
    temperature=1.0,
    device="cpu",
    max_batch_size=16,
    preprocess=True,
):
    """
    Compute jailbreak scores for a list of texts.
    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[float]: A list of jailbreak scores for each text.
    """
    return get_scores_for_texts(
        model, tokenizer, texts, [2], temperature, device, max_batch_size, preprocess
    )


def get_indirect_injection_scores_for_texts(
    model,
    tokenizer,
    texts,
    temperature=1.0,
    device="cpu",
    max_batch_size=16,
    preprocess=True,
):
    """
    Compute indirect injection scores for a list of texts.
    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        temperature (float): The temperature for the softmax function.
        device (str): The device to evaluate the model on.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[float]: A list of indirect injection scores for each text.
    """
    return get_scores_for_texts(
        model, tokenizer, texts, [1, 2], temperature, device, max_batch_size, preprocess
    )
