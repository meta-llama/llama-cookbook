# Usage: python main.py mydocument.pdf
import traceback
import os
import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# fix the annoying path bug
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.parsers import PDFParser, HTMLParser, YouTubeParser, DOCXParser, TXTParser, PPTParser

def determine_parser(file_path: str):
    if 'youtube.com' in file_path or 'youtu.be' in file_path:
        return YouTubeParser()
    if os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()

        #mapping
        parsers = {
            '.pdf': PDFParser(),
            '.html': HTMLParser(),
            '.htm': HTMLParser(),
            '.docx': DOCXParser(),
            '.pptx': PPTParser(),
            '.txt': TXTParser(),
        }
        
        if ext in parsers:
            return parsers[ext]
        else:
            supported = ", ".join(parsers.keys())
            raise ValueError(f"Can't parse {ext} files yet. Supported formats: {supported}")
    if file_path.startswith(('http://', 'https://')) and not ('youtube.com' in file_path or 'youtu.be' in file_path):
        return HTMLParser()
    if not os.path.exists(file_path):
        raise ValueError(f"File not found: {file_path}")
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def process_file(input_path, output_dir, output_filename):
    start_time = time.time()
    if output_dir is None:
        output_dir = 'data/output'
    os.makedirs(output_dir, exist_ok=True)
    if output_filename is None:
        #  load and save as .txt
        if os.path.exists(input_path):
            output_filename = os.path.splitext(os.path.basename(input_path))[0] + '.txt'
        else:
            #url becomes file name
            if 'youtube.com' in input_path or 'youtu.be' in input_path:
                if 'youtu.be/' in input_path:
                    video_id = input_path.split('/')[-1].split('?')[0]
                else:
                    video_id = ""
                    if 'v=' in input_path:
                        video_id = input_path.split('v=')[1].split('&')[0]
                    else:
                        video_id = input_path[-11:]
                        
                output_filename = f"youtube_{video_id}.txt"
            else:
                clean_url = ''.join(c if c.isalnum() else '_' for c in input_path.split('//')[-1][:30])
                output_filename = f"web_{clean_url}.txt"

    if not output_filename.endswith('.txt'):
        output_filename += '.txt'
    output_path = os.path.join(output_dir, output_filename)

    #pick parser
    parser = determine_parser(input_path)
    print(f"Parsing {input_path}...")
    try:
        content = parser.parse(input_path)
        parser.save(content, output_path)
        
        elapsed = time.time() - start_time
        size_kb = os.path.getsize(output_path) / 1024
        print(f"✓ Saved {size_kb:.1f}KB to {output_path} ({elapsed:.1f}s)")
        
        return output_path
    except Exception as e:
        print(f"Parsing failed: {str(e)}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description='Convert documents to plain text for LLM processing.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('input', help='File or URL to parse')
    parser.add_argument('-o', '--output-dir', help='Where to save the output', default='data/output')
    parser.add_argument('-n', '--name', help='Custom output filename')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show more details')
    
    args = parser.parse_args()
    
    try:
        output_path = process_file(args.input, args.output_dir, args.name)
        print(f"Saving text to....{output_path}")
        return 0
    except Exception as e:
        if args.verbose:
            traceback.print_exc()
        else:
            print(f"Error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())