"""
Utility functions for evaluation of structured data extraction.

This module provides helper functions for comparing JSON outputs,
calculating accuracy metrics, and analyzing differences between
predicted and actual structured data.
"""

import ast, json
import logging
import re
from typing import Any, Dict, List, Union

from json_repair import repair_json
from jsondiff import diff

# Setup logging
logger = logging.getLogger(__name__)

# Compile regex patterns once for better performance
JSON_BLOCK_OPEN = re.compile(r"```json")
JSON_BLOCK_CLOSE = re.compile(r"}\s+```")


def calculate_json_accuracy(
    actual: Union[str, Dict[str, Any]],
    predicted: Union[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate accuracy metrics between predicted and actual JSON.

    Args:
        actual: The ground truth JSON as string or dict
        predicted: The predicted JSON as string or dict

    Returns:
        Dict containing accuracy metrics including score, diffs, and field counts
    """
    try:
        # Use JSONUtils from src for consistency
        actual = JSONUtils.load_json_from_str(actual)
        predicted = JSONUtils.load_json_from_str(predicted)
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return {
            "score": -1,
            "full_json_diff": {},
            "json_diff": {},
            "nb_different_fields": -1,
            "total_fields": -1,
        }

    full_diff_result = diff(actual, predicted, syntax="symmetric")
    diff_result = diff(predicted, actual)
    total_fields = count_total_fields(actual)

    if not diff_result:
        return {
            "score": 1,
            "full_json_diff": {},
            "json_diff": {},
            "nb_different_fields": 0,
            "total_fields": total_fields,
        }

    changes = count_number_of_differences(diff_result)
    score = max(0, (total_fields - changes) / total_fields)
    return {
        "score": round(score, 4),
        "full_json_diff": str(full_diff_result),
        "json_diff": str(diff_result),
        "nb_different_fields": changes,
        "total_fields": total_fields,
    }


def count_number_of_differences(differences) -> int:
    """
    Count the number of differences in a JSON diff object.

    Args:
        differences: The diff object or string representation

    Returns:
        int: Total number of differences found
    """
    differences = JSONUtils.load_json_from_str(differences)

    def count_differences(differences: Any) -> int:
        count = 0
        if isinstance(differences, list) or isinstance(differences, tuple):
            count += sum([count_differences(item) for item in differences])
        if isinstance(differences, dict):
            for _, value in differences.items():
                if isinstance(value, dict):
                    # Recursively count differences in nested objects
                    count += count_differences(value)
                elif isinstance(value, list):
                    count += sum([count_differences(v) for v in value])
                else:
                    # Additions or deletions
                    count += 1
        return count

    return count_differences(differences)


def count_total_fields(obj: Any) -> int:
    """
    Count the total number of fields in a JSON object.

    Args:
        obj: The JSON object to count fields in

    Returns:
        int: Total number of fields
    """
    count = 0

    def traverse(current: Any) -> None:
        """Recursively traverse the object and count fields."""
        nonlocal count
        if not current or not isinstance(current, (dict, list)):
            return

        if isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    traverse(item)
                else:
                    count += 1
        else:
            for key, value in current.items():
                if "__" in key:
                    continue
                if isinstance(value, (str, int, float, bool)) or value is None:
                    count += 1
                elif isinstance(value, (dict, list)):
                    traverse(value)

    traverse(obj)
    return count


class JSONUtils:
    """Utility functions for working with JSON data."""

    @staticmethod
    def extract_json_blocks(content: str) -> List[str]:
        """
        Extract JSON code blocks from markdown-formatted text.

        Parses a string containing markdown-formatted text and extracts all JSON blocks
        that are enclosed in ```json ... ``` code blocks. This is useful for extracting
        structured data from LLM responses.

        Args:
            content: The markdown-formatted text containing JSON code blocks

        Returns:
            List[str]: A list of extracted JSON strings (without the markdown delimiters)
        """
        blocs_ix = []
        str_ptr = 0

        while str_ptr < len(content):
            start_ix = content.find("```json", str_ptr)
            if start_ix == -1:
                break
            start_ix += len("```json")
            end_match = JSON_BLOCK_CLOSE.search(content[start_ix:])
            if end_match:
                end_ix = start_ix + end_match.start() + 1
            else:
                end_ix = len(content)  # no closing tag, take the rest of the string
            blocs_ix.append((start_ix, end_ix))
            str_ptr = end_ix + 1

        return [content[ix[0] : ix[1]].strip() for ix in blocs_ix]

    @staticmethod
    def load_json_from_str(json_str: str) -> Dict[str, Any]:
        """
        Parse a JSON string into a Python dictionary.

        Attempts to parse a string as JSON using multiple methods. First tries standard
        json.loads(), then falls back to ast.literal_eval() if that fails. This provides
        more robust JSON parsing for LLM outputs that might not be perfectly formatted.

        Args:
            json_str: The JSON string to parse

        Returns:
            Dict[str, Any]: The parsed JSON as a dictionary

        Raises:
            ValueError: If parsing fails
        """
        if not isinstance(json_str, str):
            return json_str

        json_str = repair_json(json_str)
        try:
            return json.loads(json_str)
        except json.decoder.JSONDecodeError:
            # Try with None replacement
            json_str = json_str.replace("null", "None")
            try:
                return ast.literal_eval(json_str)
            except:
                raise ValueError(f"Failed to load valid JSON from string: {json_str}")

    @staticmethod
    def extract_json_from_response(content: str) -> Dict[str, Any]:
        """
        Extract and parse JSON from an LLM response.

        Processes a response from an LLM that may contain JSON in a markdown code block.
        First checks if the response contains markdown-formatted JSON blocks and extracts them,
        then parses the JSON string into a Python dictionary.

        Args:
            content: The LLM response text that may contain JSON

        Returns:
            Dict[str, Any]: The parsed JSON as a dictionary

        Raises:
            ValueError: If extraction or parsing fails
        """
        try:
            if "```json" in content:
                json_blocks = JSONUtils.extract_json_blocks(content)
                if not json_blocks:
                    raise ValueError("No JSON blocks found in response")
                content = json_blocks[-1]

            return JSONUtils.load_json_from_str(content)
        except Exception as e:
            raise ValueError(f"Failed to extract JSON from response: {str(e)}")
