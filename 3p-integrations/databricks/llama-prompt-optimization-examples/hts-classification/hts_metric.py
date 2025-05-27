"""
HTS Code format validation metric.

This module provides a metric for evaluating if HTS code predictions
exactly match the format of the ground truth (must use same format - numeric or dotted).
If gold uses numeric format (XXXXXXXXXX), prediction must also use numeric format.
If gold uses dotted format (XX.XX.XX.XXXX), prediction must also use dotted format.
"""

import re
from typing import Any, Dict, Union
from llama_prompt_ops.core.metrics import MetricBase

class HTSCodeMetric(MetricBase):
    """
    Metric for evaluating HTS code format predictions.
    
    This metric verifies that:
    1. The prediction is a valid 10-digit HTS code
    2. The prediction uses EXACTLY the same format as the ground truth:
       - If gold is XXXXXXXXXX, prediction must be XXXXXXXXXX
       - If gold is XX.XX.XX.XXXX, prediction must be XX.XX.XX.XXXX
    """
    
    def __init__(self, output_field: str = "hts_code"):
        """
        Initialize the HTS code metric.
        
        Args:
            output_field: Field name containing the HTS code in the prediction
        """
        self.output_field = output_field
        
    def __call__(
        self, 
        gold: Any, 
        pred: Any, 
        trace: bool = False,
        **kwargs
    ) -> Union[Dict[str, float], float]:
        """
        Evaluate if prediction format exactly matches the ground truth HTS code format.
        Returns 0.0 if formats don't match (e.g., numeric vs dotted).
        
        Args:
            gold: Ground truth example
            pred: Predicted example
            trace: Whether to return detailed results
            
        Returns:
            Score (1.0 for exact format match, 0.0 otherwise) or detailed results dict
        """
        # Extract values
        gold_value = self._extract_value(gold)
        pred_value = self._extract_value(pred)
        
        if trace:
            print(f"Gold value: {gold_value}")
            print(f"Predicted value: {pred_value}")
        
        # Clean the values
        gold_clean = self._clean_hts_code(gold_value)
        pred_clean = self._clean_hts_code(pred_value)
        
        # Get format patterns
        gold_format = self._get_format_pattern(gold_clean)
        pred_format = self._get_format_pattern(pred_clean)
        
        # Check if prediction is valid and matches gold format exactly
        is_valid = (
            gold_format != 'invalid' and  # Gold must be valid
            pred_format != 'invalid' and  # Prediction must be valid
            gold_format == pred_format    # Formats must match exactly
        )
        
        if trace:
            return {
                "score": 1.0 if is_valid else 0.0,
                "gold_format": gold_format,
                "pred_format": pred_format,
                "formats_match": gold_format == pred_format,
                "cleaned_gold": gold_clean,
                "cleaned_pred": pred_clean,
                "explanation": self._get_explanation(gold_format, pred_format)
            }
        
        return 1.0 if is_valid else 0.0
    
    def _extract_value(self, data: Any) -> str:
        """
        Extract HTS code value from various input types.
        """
        if isinstance(data, str):
            return data
        
        if isinstance(data, dict):
            return str(data.get(self.output_field, ""))
            
        if hasattr(data, self.output_field):
            return str(getattr(data, self.output_field))
            
        if hasattr(data, "outputs") and isinstance(data.outputs, dict):
            return str(data.outputs.get(self.output_field, ""))
            
        return str(data)
    
    def _clean_hts_code(self, code: str) -> str:
        """
        Clean HTS code by removing spaces and standardizing format.
        """
        # Remove all whitespace
        code = "".join(code.split())
        
        # Remove any non-alphanumeric characters except dots
        code = re.sub(r'[^0-9.]', '', code)
        
        return code
    
    def _get_format_pattern(self, code: str) -> str:
        """
        Get the format pattern of the code.
        Returns:
        - 'numeric' for XXXXXXXXXX format
        - 'dotted' for XX.XX.XX.XXXX format
        - 'invalid' for any other format
        """
        if re.match(r'^\d{10}$', code):
            return 'numeric'
        elif re.match(r'^\d{2}\.\d{2}\.\d{2}\.\d{4}$', code):
            return 'dotted'
        else:
            return 'invalid'
    
    def _get_explanation(self, gold_format: str, pred_format: str) -> str:
        """
        Get a human-readable explanation of the format comparison.
        """
        if gold_format == 'invalid':
            return "Gold standard format is invalid"
        if pred_format == 'invalid':
            return "Prediction format is invalid"
        if gold_format != pred_format:
            return f"Format mismatch: Gold uses {gold_format} format but prediction uses {pred_format} format"
        return "Formats match correctly"