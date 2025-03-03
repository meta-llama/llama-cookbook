import os
import json
import re
import time
import subprocess
import sys
from cerebras.cloud.sdk import Cerebras
from typing import Dict, List, Any, Optional, Tuple
    
class QAGenerator:
    def __init__(self, api_key=None, model="llama-3.3-70b"):
        if api_key is None:
            api_key = os.environ.get("CEREBRAS_API_KEY")
            if not api_key:
                raise ValueError("Set Key")
        self.model = model
        self.client = Cerebras(api_key=api_key)
    
    def generate_summary(self, document_text):
        print(f"Document summarising...")
        prompt = """You are summarizing a document for LLM training data preparation.
        
Write a comprehensive summary that captures:
1. The document's title and main topic
2. Key concepts and terminology
3. Main arguments or findings
4. Notable data points or examples
5. Overall structure and purpose

Your summary should be detailed enough that it could be used to understand what kinds of questions can be asked about this document. Focus on factual information rather than opinions."""
        
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": document_text}
            ],
            model=self.model,
            temperature=0.1,
        )
        
        summary = response.choices[0].message.content
        print(f"Summary generated ({len(summary)} chars)")
        return summary
    
    def generate_qa_pairs(self, document_text, summary, chunk_size=4000, num_pairs=25):
        print(f"Generating QA pairs")
        chunks = self._split_into_chunks(document_text, chunk_size)
        print(f"Document split into {len(chunks)} chunks")
        all_qa_pairs = []
        pairs_per_chunk = max(1, round(num_pairs / len(chunks)))
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} (target: ~{pairs_per_chunk} pairs)")
            prompt = f"""You are creating question-answer pairs for fine-tuning LLMs.

Below is a chunk of text from a document about: {summary[:100]}...

Your task is to create {pairs_per_chunk} high-quality question-answer pairs based ONLY on the information in this text chunk.

For each pair:
1. Create a specific, clear question about important information
2. Provide a comprehensive but concise answer
3. Focus on technical details and specific facts
4. Ensure answers are directly supported by the text

Return ONLY valid JSON formatted as:
[
  {{
    "question": "Detailed question about the content?",
    "answer": "Comprehensive answer based on the text."
  }},
  ...
]

Here is the text chunk:
---
{chunk}
---"""
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": prompt}
                ],
                model=self.model,
                temperature=0.7,
            )
            try:
                qa_text = response.choices[0].message.content
                chunk_pairs = self._extract_qa_pairs(qa_text)
                all_qa_pairs.extend(chunk_pairs)
                print(f"  Generated {len(chunk_pairs)} pairs from chunk {i+1}")
            except Exception as e:
                print(f"  Error processing chunk {i+1}: {str(e)}")
        
        print(f"Generated {len(all_qa_pairs)} QA pairs")
        return all_qa_pairs
    
    def rate_qa_pairs(self, qa_pairs, summary, threshold=7.0):
        if not qa_pairs:
            return [], {"total": 0, "filtered": 0, "retention_rate": 0, "avg_score": 0}
        
        print(f"Evaluating pairs {len(qa_pairs)}")
        batch_size = 8
        batches = [qa_pairs[i:i+batch_size] for i in range(0, len(qa_pairs), batch_size)]
        
        rated_pairs = []
        total_score = 0
        for i, batch in enumerate(batches):
            print(f"Rating batch {i+1}/{len(batches)}...")
            batch_json = json.dumps(batch, indent=2)
            prompt = f"""Rate these question-answer pairs for LLM fine-tuning on a scale of 1-10.

Here is a summary of the document these pairs are based on:
---
{summary[:500]}...
---

For each pair below, rate its quality on a scale of 1-10 based on:
1. Relevance to the document topic
2. Question specificity and clarity
3. Answer accuracy and completeness
4. Overall educational value

The pairs to evaluate:
{batch_json}

IMPORTANT: Return ONLY a JSON array containing each original pair with an added "rating" field.
Use this exact format:
[
  {{"question": "...", "answer": "...", "rating": 8}},
  ...
]"""
            
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": prompt}
                    ],
                    model=self.model,
                    temperature=0.0
                )
                
                rated_batch = self._extract_rated_pairs(response.choices[0].message.content)
                for pair in rated_batch:
                    if "rating" in pair:
                        total_score += pair["rating"]
                        if pair["rating"] >= threshold:
                            rated_pairs.append(pair)
            
            except Exception as e:
                print(f"Error rating batch {i+1}: {str(e)}")
            time.sleep(0.5)
        
        metrics = {
            "total": len(qa_pairs),
            "filtered": len(rated_pairs),
            "retention_rate": round(len(rated_pairs) / len(qa_pairs), 2) if qa_pairs else 0,
            "avg_score": round(total_score / len(qa_pairs), 1) if qa_pairs else 0
        }
        
        print(f"Keeping {len(rated_pairs)} out of {len(qa_pairs)} pairs")
        return rated_pairs, metrics
    
    def _split_into_chunks(self, text, chunk_size):
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk)
                current_chunk = para
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _extract_qa_pairs(self, text):
        text = text.strip()
        if text.startswith('[') and text.endswith(']'):
            try:
                return json.loads(text)
            except:
                pass       
        # annoying```json start fix
        if '```' in text:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass
        return []
    
    def _extract_rated_pairs(self, text):
        array_pattern = r'\[\s*\{\s*"question".*"rating".*\}\s*\]'
        array_match = re.search(array_pattern, text)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except:
                pass
        
        #annoying ``json response fix
        if '```' in text:
            code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_match:
                code_content = code_match.group(1).strip()
                if code_content.startswith('[') and code_content.endswith(']'):
                    try:
                        return json.loads(code_content)
                    except:
                        pass

        print("Using regex to extract rating pairs")
        pair_pattern = r'{\s*"question":\s*"([^"]*)"\s*,\s*"answer":\s*"([^"]*)"\s*,\s*"rating":\s*(\d+)'
        pairs = []
        for match in re.finditer(pair_pattern, text):
            try:
                q = match.group(1).replace('\\"', '"')
                a = match.group(2).replace('\\"', '"')
                r = int(match.group(3))
                pairs.append({"question": q, "answer": a, "rating": r})
            except:
                pass
        if pairs:
            return pairs
        return []
    
    def process_document(self, document_text, num_pairs=25, quality_threshold=7.0):
        summary = self.generate_summary(document_text)
        qa_pairs = self.generate_qa_pairs(document_text, summary, num_pairs=num_pairs)
        filtered_pairs, metrics = self.rate_qa_pairs(qa_pairs, summary, threshold=quality_threshold)
        conversations = self._convert_to_conversations(filtered_pairs)
        result = {
            "summary": summary,
            "qa_pairs": qa_pairs,
            "filtered_pairs": filtered_pairs,  
            "conversations": conversations, 
            "metrics": metrics  
        }
        
        return result
    
    def _convert_to_conversations(self, qa_pairs):
        conversations = []
        
        for pair in qa_pairs:
            conversation = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that provides accurate, detailed responses."
                },
                {
                    "role": "user",
                    "content": pair["question"]
                },
                {
                    "role": "assistant",
                    "content": pair["answer"]
                }
            ]
            conversations.append(conversation)
        
        return conversations