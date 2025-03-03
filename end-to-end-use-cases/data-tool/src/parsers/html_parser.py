import os
import re
import requests
from bs4 import BeautifulSoup
import time

class HTMLParser:
    def __init__(self):
        # usual headers
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
        }
    
    def parse(self, source):
        if os.path.exists(source):
            with open(source, 'r', encoding='utf-8') as f:
                html_content = f.read()
        elif source.startswith(('http://', 'https://')):
            try:
                session = requests.Session()
                adapter = requests.adapters.HTTPAdapter(
                    max_retries=3,
                    pool_connections=10,
                    pool_maxsize=10
                )
                session.mount('http://', adapter)
                session.mount('https://', adapter)

                response = session.get(
                    source, 
                    headers=self.headers, 
                    timeout=20,
                    allow_redirects=True
                )
                response.raise_for_status()
                html_content = response.text
            except requests.exceptions.RequestException as e:
                raise ValueError(f"Failed to fetch URL: {e}")
        else:
            raise ValueError(f"Invalid source: {source}. Must be a file path or URL.")
        soup = BeautifulSoup(html_content, 'html.parser')
        
        title = ""
        if soup.title:
            title = f"Title: {soup.title.string.strip()}\n\n"
        
        # remove garbage
        for element in soup(['script', 'style', 'head', 'meta', 'noscript', 'svg', 
                           'header', 'footer', 'nav', 'aside']):
            element.extract()
        
        main_content = ""
        
        main_elements = soup.select('main, article, .content, #content, .main, #main')
        if main_elements:
            main_content = self._extract_text_with_structure(main_elements[0])
        else:
            body = soup.find('body')
            if body:
                main_content = self._extract_text_with_structure(body)
            else:
                main_content = soup.get_text(separator='\n')
                
        cleaned_text = self._clean_text(main_content)
        
        return title + cleaned_text
    
    def _extract_text_with_structure(self, element):
        result = []
        
        for heading in element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            heading_text = heading.get_text(strip=True)
            if heading_text:
                heading_level = int(heading.name[1])
                prefix = '#' * heading_level + ' '
                result.append(f"\n{prefix}{heading_text}\n")
                heading.extract()
        
        for para in element.find_all('p'):
            para_text = para.get_text(strip=True)
            if para_text:
                result.append(para_text + '\n')
        
        for ul in element.find_all(['ul', 'ol']):
            for li in ul.find_all('li'):
                li_text = li.get_text(strip=True)
                if li_text:
                    result.append(f"• {li_text}")
            result.append('')
        
        # If we haven't captured any structured content, fallback to regular text
        if not result:
            text = element.get_text(separator='\n', strip=True)
            result = [text]
            
        return '\n'.join(result)
    
    def _clean_text(self, text):
        lines = [line.strip() for line in text.splitlines()]
        non_empty_lines = [line for line in lines if line]
        unique_lines = []
        prev_line = None
        for line in non_empty_lines:
            if line != prev_line:
                unique_lines.append(line)
                prev_line = line
                
        return '\n'.join(unique_lines)
    
    def save(self, content, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)