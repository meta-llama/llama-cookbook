# This script eats the parsed texts and gives QA pairs
import os
import json
import argparse
import sys
import traceback
from pathlib import Path

# Annoying path bug fix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import QAGenerator

def process_file(file_path, output_dir=None, model="llama-3.3-70b", 
                api_key=None, text_file=None, num_pairs=25, threshold=7.0):
    if output_dir is None:
        output_dir = 'data/qa_pairs'
    os.makedirs(output_dir, exist_ok=True)
    if text_file and os.path.exists(text_file):
        print(f"Using pre-processed text from {text_file}")
        with open(text_file, 'r', encoding='utf-8') as f:
            document_text = f.read()
    else:
        from src.main import process_file as parse_file
        print(f"Parsing {file_path} to extract text...")
        parsed_path = parse_file(file_path)
        
        print(f"Reading parsed content from {parsed_path}")
        with open(parsed_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
    generator = QAGenerator(api_key=api_key, model=model)
    
    result = generator.process_document(document_text,num_pairs=num_pairs,quality_threshold=threshold)
    base_name = os.path.basename(file_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{base_name}_qa_pairs.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print("Writing summary")
    print(f"QA Pairs Generated: {result['metrics']['total']}")
    print(f"Quality Pairs: {result['metrics']['filtered']} ({result['metrics']['retention_rate']*100:.1f}%)")
    return output_path

def main():
    parser = argparse.ArgumentParser(
        description='Generate QA pairs from documents using Llama',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input', help='File to parse and generate QA pairs from')
    parser.add_argument('-o', '--output-dir', help='Where to save the output', default='data/qa_pairs')
    parser.add_argument('-m', '--model', help='Model to use', default='llama-3.3-70b')
    parser.add_argument('-k', '--api-key', help='API key (defaults to env var)')
    parser.add_argument('-t', '--text-file', help='Path to already parsed text file (skips parsing step)')
    parser.add_argument('-n', '--num-pairs', type=int, help='Target number of QA pairs to generate', default=25)
    parser.add_argument('--threshold', type=float, help='Quality threshold for filtering pairs (1-10)', default=7.0)
    
    args = parser.parse_args()
    
    try:
        output_path = process_file(
            args.input,
            args.output_dir,
            args.model,
            args.api_key,
            args.text_file,
            args.num_pairs,
            args.threshold
        )
        print(f"QA pairs saved to {output_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())