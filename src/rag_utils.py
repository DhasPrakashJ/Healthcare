import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

# Load embeddings model
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

def build_faiss_index(texts):
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings

def retrieve_similar(query, texts, index, embeddings, top_k=3):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [(texts[i], distances[0][idx]) for idx, i in enumerate(indices[0])]

# Simple LLM answer generation
def generate_response(prompt, context=""):
    full_prompt = f"{context}\nUser: {prompt}\nAI:"
    llm = pipeline("text-generation", model="gpt2")
    response = llm(full_prompt, max_length=200, do_sample=True)[0]['generated_text']
    return response.split("AI:")[-1].strip()
