import os
import requests
import json
import time
import io
import re
import gradio as gr
import PyPDF2
from together import Together



def download_pdf(url, save_path=None):
    if url is None or 'arxiv.org' not in url:
        return None
    response = requests.get(url)
    if save_path:
        with open(save_path, 'wb') as f:
            f.write(response.content)
    return response.content

def extract_arxiv_pdf_url(arxiv_url):
    # Check if URL is already in PDF format
    if 'arxiv.org/pdf/' in arxiv_url:
        return arxiv_url
    
    # Extract arxiv_id from different URL formats
    arxiv_id = None
    if 'arxiv.org/abs/' in arxiv_url:
        arxiv_id = arxiv_url.split('arxiv.org/abs/')[1].split()[0]
    elif 'arxiv.org/html/' in arxiv_url:
        arxiv_id = arxiv_url.split('arxiv.org/html/')[1].split()[0]
    
    if arxiv_id:
        return f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    return None  # Return None if no valid arxiv_id found

def extract_text_from_pdf(pdf_content):
    pdf_file = io.BytesIO(pdf_content)
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_references_with_llm(pdf_content):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_content)
    
    # Truncate if too long
    max_length = 50000
    if len(text) > max_length:
        text = text[:max_length] + "..."

    client = Together(api_key="4a9f7614927a3abd614a170fc5d4283b68e8685764c06dd98b1e1066651b9229")

    citations = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {
                "role":"user",
                "content":f"Extract all the arXiv citations from Reference section of the paper including their title, authors and origins. Paper: {text} "
            }
        ],
        temperature=0.3,
    )
    
    # Prepare prompt for Llama 4
    prompt = f"""   
                Extract the arXiv ID from the list of citations provided, including preprint arXiv ID. If there is no arXiv ID presented with the list, skip that citations.
                
                Here are some examples on arXiv ID format:
                1. arXiv preprint arXiv:1607.06450, where 1607.06450 is the arXiv ID.
                2. CoRR, abs/1409.0473, where 1409.0473 is the arXiv ID.

                Then, return a JSON array of objects with 'title' and 'ID' fields strictly in the following format, only return the paper title if it's arXiv ID is extracted:

                Output format: [{{\"title\": \"Paper Title\", \"ID\": \"arXiv ID\"}}]

                DO NOT return any other text.

                List of citations:
                {citations}
                """

    
    response = client.chat.completions.create(
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        messages=[
            {
                "role":"user",
                "content":prompt
            }
        ],
        temperature=0.3,
    )
    response_json = response.choices[0].message.content

    # Convert the JSON string to a Python object
    references = []
    try:
        references = json.loads(response_json)
     # Now you can work with `references` as a Python list or dictionary
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")

    return references

# Check if ref_id is a valid arXiv ID
def is_valid_arxiv_id(ref_id):
    # arXiv IDs are typically in the format of "1234.56789" or "1234567"
    return bool(re.match(r'^\d{4}\.\d{4,5}$', ref_id) or re.match(r'^\d{7}$', ref_id))

def download_arxiv_paper_and_citations(arxiv_url, download_dir, progress=None):
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    if progress:
        progress("Downloading main paper PDF...")
    
    # Download main paper PDF
    pdf_url = extract_arxiv_pdf_url(arxiv_url)
    main_pdf_path = os.path.join(download_dir, 'main_paper.pdf')
    main_pdf_content = download_pdf(pdf_url, main_pdf_path)
    
    if main_pdf_content is None:
        if progress:
            progress("Invalid Url. Valid example: https://arxiv.org/abs/1706.03762v7")
        return None, 0

    if progress:
        progress("Main paper downloaded. Extracting references...")
    
    # Extract references using LLM
    references = extract_references_with_llm(main_pdf_content)

    if progress:
        progress(f"Found {len(references)} references. Downloading...")
        time.sleep(1)
    
    # Download reference PDFs
    all_pdf_paths = [main_pdf_path]
    for i, reference in enumerate(references):
        ref_title = reference.get("title")
        ref_id = reference.get("ID")
        if ref_id and is_valid_arxiv_id(ref_id):
            ref_url = f'https://arxiv.org/pdf/{ref_id}'
            ref_pdf_path = os.path.join(download_dir, f'{ref_title}.pdf')
            if progress:
                progress(f"Downloading reference {i+1}/{len(references)}...{ref_title}")
                time.sleep(0.2)
            try:
                download_pdf(ref_url, ref_pdf_path)
                all_pdf_paths.append(ref_pdf_path)
            except Exception as e:
                if progress:
                    progress(f"Error downloading {ref_url}: {str(e)}")
                    time.sleep(0.2)
    
    # Create a list of all PDF paths
    paths_file = os.path.join(download_dir, 'pdf_paths.txt')
    with open(paths_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_pdf_paths))
    
    if progress:
        progress(f"All papers downloaded. Total references: {len(references)}")
        time.sleep(1)
    return paths_file, len(references)

def ingest_paper_with_llama(paths_file, progress=None):
    total_text = ""
    total_word_count = 0

    if progress:
        progress("Ingesting paper content...")

    with open(paths_file, 'r', encoding='utf-8') as f:
        pdf_paths = f.read().splitlines()
        

    for i, pdf_path in enumerate(pdf_paths):
        if progress:
            progress(f"Ingesting PDF {i+1}/{len(pdf_paths)}...")
        with open(pdf_path, 'rb') as pdf_file:
            pdf_content = pdf_file.read()
            text = extract_text_from_pdf(pdf_content)
            total_text += text + "\n\n"
            total_word_count += len(text.split())

    if progress:
        progress("Paper ingestion complete!")

    return total_text, total_word_count
    
def gradio_interface():
    paper_content = {"text": ""}
    
    def process(arxiv_url, progress=gr.Progress()):
        download_dir = 'downloads'
        progress(0, "Starting download...")
        paper_path, num_references = download_arxiv_paper_and_citations(arxiv_url, download_dir, 
                                                      lambda msg: progress(0.3, msg))
        if paper_path is None:
            return "Invalid Url. Valid example: https://arxiv.org/abs/1706.03762v7"
            
        paper_content["text"], total_word_count = ingest_paper_with_llama(paper_path, 
                                                      lambda msg: progress(0.7, msg))
        progress(1.0, "Ready for chat!")
        return f"Total {total_word_count} words and {num_references} reference ingested. You can now chat about the paper and citations."

    def respond(message, history):
        user_message = message

        if not user_message:
            return history, ""
        
        # Append user message immediately
        history.append([user_message, ""])
        

        client = Together(api_key="4a9f7614927a3abd614a170fc5d4283b68e8685764c06dd98b1e1066651b9229")

        # Prepare the system prompt and user message


        system_prompt = f"""
                    You are a research assistant that have access to the paper reference below.
                    Answer questions based on your knowledge on these references.
                    If you do not know the answer, say you don't know.
                    paper reference: {paper_content["text"]}
                    """
        
        stream = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            stream=True  # Enable streaming
        )
        
        # Initialize an empty response
        full_response = ""
        
        # Stream the response chunks
        for chunk in stream:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                # Update the last message in history with the current response
                history[-1][1] = full_response
                yield history,""
        

    
    def clear_chat_history():
        return [], ""
    
    with gr.Blocks(css=".orange-button {background-color: #FF7C00 !important; color: white;}") as demo:
        gr.Markdown("# Research Analyzer")
        with gr.Column():
            input_text = gr.Textbox(label="ArXiv URL")
            status_text = gr.Textbox(label="Status", interactive=False)
            submit_btn = gr.Button("Ingest", elem_classes="orange-button")
            submit_btn.click(fn=process, inputs=input_text, outputs=status_text)
            
            gr.Markdown("## Chat with Llama")
            chatbot = gr.Chatbot()
        with gr.Row():
            msg = gr.Textbox(label="Ask about the paper", scale=5)
            submit_chat_btn = gr.Button("➤", elem_classes="orange-button", scale=1)
            
        submit_chat_btn.click(respond, [msg, chatbot], [chatbot, msg])
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
            
        def copy_last_response(history):
            if history and len(history) > 0:
                last_response = history[-1][1]
                return gr.update(value=last_response)
            return gr.update(value="No response to copy")
        
    
    demo.launch()


if __name__ == "__main__":
    gradio_interface()
