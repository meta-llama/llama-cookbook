from typing import List, Tuple

import torch
from torch.nn.functional import softmax
from transformers import AutoModelForSequenceClassification, AutoTokenizer

"""
Utilities for loading the PromptGuard model and evaluating text for jailbreaking techniques.

NOTE: this code is for PromptGuard 2. For our older PromptGuard 1 model, see prompt_guard_1_inference.py

Note that the underlying model has a maximum recommended input size of 512 tokens as a DeBERTa model.
The final two functions in this file implement efficient parallel batched evaluation of the model on a list
of input strings of arbitrary length, with the final score for each input being the maximum score across all
chunks of the input string.
"""

MAX_TOKENS = 512
DEFAULT_BATCH_SIZE = 16
DEFAULT_TEMPERATURE = 1.0
DEFAULT_DEVICE = "cpu"
DEFAULT_MODEL_NAME = "meta-llama/Llama-Prompt-Guard-2-86M"


def load_model_and_tokenizer(
    model_name: str = "meta-llama/Prompt-Guard-2-86M", device: str = DEFAULT_DEVICE
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer, str]:
    """
    Load the PromptGuard model and tokenizer, and move the model to the specified device.

    Args:
        model_name (str): The name of the model to load.
        device (str): The device to load the model on. If None, it will use CUDA if available, else CPU.

    Returns:
        tuple: The loaded model, tokenizer, and the device used.
    """
    try:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer, device
    except Exception as e:
        raise RuntimeError(f"Failed to load model and tokenizer: {str(e)}")


def get_class_scores(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """
    Evaluate the model on the given text with temperature-adjusted softmax.
    Note, as this is a DeBERTa model, the input text should have a maximum length of 512.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        text (str): The input text to classify.
        temperature (float): The temperature for the softmax function. Default is 1.0.

    Returns:
        torch.Tensor: The scores for each class adjusted by the temperature.
    """

    # Get the device from the model
    device = next(model.parameters()).device

    # Encode the text
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Get logits from the model
    with torch.no_grad():
        logits = model(**inputs).logits
    # Apply temperature scaling
    scaled_logits = logits / temperature
    # Apply softmax to get scores
    scores = softmax(scaled_logits, dim=-1)
    return scores


def get_jailbreak_score(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    text: str,
    temperature: float = DEFAULT_TEMPERATURE,
) -> float:
    """
    Evaluate the probability that a given string contains malicious jailbreak or prompt injection.
    Appropriate for filtering dialogue between a user and an LLM.

    Args:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
        text (str): The input text to evaluate.
        temperature (float): The temperature for the softmax function. Default is 1.0.

    Returns:
        float: The probability of the text containing malicious content.
    """
    probabilities = get_class_scores(model, tokenizer, text, temperature)
    return probabilities[0, 1].item()


def process_text_batch(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    temperature: float = DEFAULT_TEMPERATURE,
) -> torch.Tensor:
    """
    Process a batch of texts and return their class probabilities.
    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to process.
        temperature (float): The temperature for the softmax function.

    Returns:
        torch.Tensor: A tensor containing the class probabilities for each text in the batch.
    """
    # Get the device from the model
    device = next(model.parameters()).device

    # encode the texts
    inputs = tokenizer(
        texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_TOKENS
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
    scaled_logits = logits / temperature
    probabilities = softmax(scaled_logits, dim=-1)
    return probabilities


def get_scores_for_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    score_indices: List[int],
    temperature: float = DEFAULT_TEMPERATURE,
    max_batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[float]:
    """
    Compute scores for a list of texts, handling texts of arbitrary length by breaking them into chunks and processing in parallel.
    The final score for each text is the maximum score across all chunks of the text.

    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        score_indices (list[int]): Indices of scores to sum for final score calculation.
        temperature (float): The temperature for the softmax function.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[float]: A list of scores for each text.
    """
    all_chunks = []
    text_indices = []
    for index, text in enumerate(texts):
        # Tokenize the text and split into chunks of MAX_TOKENS
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
        chunks = [tokens[i : i + MAX_TOKENS] for i in range(0, len(tokens), MAX_TOKENS)]
        all_chunks.extend(chunks)
        text_indices.extend([index] * len(chunks))
    all_scores = [0.0] * len(texts)
    for i in range(0, len(all_chunks), max_batch_size):
        batch_chunks = all_chunks[i : i + max_batch_size]
        batch_indices = text_indices[i : i + max_batch_size]
        # Decode the token chunks back to text
        batch_texts = [
            tokenizer.decode(chunk, skip_special_tokens=True) for chunk in batch_chunks
        ]
        probabilities = process_text_batch(model, tokenizer, batch_texts, temperature)
        scores = probabilities[:, score_indices].sum(dim=1).tolist()
        for idx, score in zip(batch_indices, scores):
            all_scores[idx] = max(all_scores[idx], score)
    return all_scores


def get_jailbreak_scores_for_texts(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    texts: List[str],
    temperature: float = DEFAULT_TEMPERATURE,
    max_batch_size: int = DEFAULT_BATCH_SIZE,
) -> List[float]:
    """
    Compute jailbreak scores for a list of texts.
    Args:
        model (transformers.PreTrainedModel): The loaded model.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the model.
        texts (list[str]): A list of texts to evaluate.
        temperature (float): The temperature for the softmax function.
        max_batch_size (int): The maximum number of text chunks to process in a single batch.

    Returns:
        list[float]: A list of jailbreak scores for each text.
    """
    return get_scores_for_texts(
        model, tokenizer, texts, [1], temperature, max_batch_size
    )
