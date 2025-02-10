import json
import os
from typing import Dict, Any
from pathlib import Path
import ast

def clean_json_string(json_str: str) -> str:
    """Clean a JSON string by removing extra escapes."""
    # First try to parse it as a raw string literal
    try:
        # This handles cases where the string is like "\\n" -> "\n"
        cleaned = ast.literal_eval(f"'''{json_str}'''")
        return cleaned
    except:
        return json_str

def process_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively process dictionary values to clean strings."""
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            cleaned_data[key] = clean_json_string(value)
        elif isinstance(value, dict):
            cleaned_data[key] = process_dict(value)
        elif isinstance(value, list):
            cleaned_data[key] = [
                process_dict(item) if isinstance(item, dict)
                else clean_json_string(item) if isinstance(item, str)
                else item
                for item in value
            ]
        else:
            cleaned_data[key] = value
    return cleaned_data

def process_dataset(input_path: str, output_path: str):
    """Process a dataset file or directory and save cleaned version."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Process single file
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Clean the data
        if isinstance(data, dict):
            cleaned_data = process_dict(data)
        elif isinstance(data, list):
            cleaned_data = [
                process_dict(item) if isinstance(item, dict)
                else clean_json_string(item) if isinstance(item, str)
                else item
                for item in data
            ]
        
        # Write cleaned data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
            
    elif input_path.is_dir():
        # Process directory of files
        output_path.mkdir(parents=True, exist_ok=True)
        for file_path in input_path.glob('**/*.json'):
            # Maintain directory structure in output
            relative_path = file_path.relative_to(input_path)
            output_file = output_path / relative_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Process individual file
            process_dataset(str(file_path), str(output_file))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean JSON dataset by removing extra escapes')
    parser.add_argument('--input', required=True, help='Path to input JSON file or directory')
    parser.add_argument('--output', required=True, help='Path for cleaned output')
    
    args = parser.parse_args()
    process_dataset(args.input, args.output)