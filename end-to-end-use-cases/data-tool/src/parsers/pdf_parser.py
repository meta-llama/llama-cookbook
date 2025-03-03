import os
import re
from PyPDF2 import PdfReader

class PDFParser:
    def __init__(self):
        self.page_count = 0
        
    def parse(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Hey, can't find that PDF: {file_path}")
        
        # PyPDF2 does the heavy lifting    
        reader = PdfReader(file_path)
        self.page_count = len(reader.pages)
        chunks = []
        
        # Sometimes PDFs have garbage characters, so we'll clean as we go
        for i, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text()
                # Skip mostly empty pages
                if page_text and len(page_text.strip()) > 20:
                    # some files have junk spaces, this skils that
                    clean_text = self._clean_text(page_text)
                    chunks.append(clean_text)
            except Exception as e:
                # Sometimes one bad page will interrupt the parsing, this fixes
                chunks.append(f"[Error extracting page {i+1}: {str(e)}]")
                
        return "\n\n".join(chunks)
    
    def _clean_text(self, text):
        # Fix sharp edges
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'([.!?]) +', r'\1\n', text)  # Add proper line breaks
        return text.strip()
            
    def save(self, content, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"# Extracted from PDF ({self.page_count} pages)\n\n")
            f.write(content)
            
        return os.path.getsize(output_path)