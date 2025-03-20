import torch
from transformers import AutoTokenizer, AutoModel
from llama_index.core.base.embeddings.base import BaseEmbedding

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
model_id = "jinaai/jina-embeddings-v2-base-en" #"jinaai/jina-embeddings-v3"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)

# Define function to generate embeddings
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)    
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy() #.to(torch.float32)


class LocalJinaEmbedding(BaseEmbedding):
    def __init__(self):
        super().__init__()

    def _get_text_embedding(self, text):
        return get_embedding(text).tolist()  # Ensure compatibility with LlamaIndex

    def _get_query_embedding(self, query):
        return get_embedding(query).tolist()
    
    async def _aget_query_embedding(self, query: str) -> list:
        return get_embedding(query).tolist()



def test(): #this did not produce reasonable results for some reason
    #!pip install llama-index-embeddings-huggingface
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
    embed_model = HuggingFaceEmbedding(model_name=model_id)


if __name__=="__main__":
	emb = get_embedding("hi there")
	print(emb.shape)
