# Most straight forward-copy paste :)
import os

class TXTParser:
    def __init__(self):
        pass
    
    def parse(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return content
    
    def save(self, content, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)