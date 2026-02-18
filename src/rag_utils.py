import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import torch


# ----------------------------------
# Model Initialization (Load Once)
# ----------------------------------

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

# Embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# LLM pipeline (loaded once)
llm = pipeline(
    "text-generation",
    model="gpt2",
    device=0 if device == "cuda" else -1
)


# ----------------------------------
# Build FAISS Index (Cosine Similarity)
# ----------------------------------

def build_faiss_index(texts):
    embeddings = embed_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    # Cosine similarity using Inner Product
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index, embeddings


# ----------------------------------
# Retrieve Top-K Similar Documents
# ----------------------------------

def retrieve_similar(query, texts, index, embeddings, top_k=3):
    query_vec = embed_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(query_vec, top_k)

    results = []
    for rank, idx in enumerate(indices[0]):
        results.append((texts[idx], float(scores[0][rank])))

    return results


# ----------------------------------
# Generate AI Response
# ----------------------------------

def generate_response(prompt, context=""):
    system_instruction = (
        "You are a supportive AI mental health companion. "
        "Provide empathetic, calm, and helpful responses. "
        "Do not provide medical diagnosis. Encourage professional help if needed.\n"
    )

    full_prompt = f"{system_instruction}\nContext:\n{context}\n\nUser: {prompt}\nAI:"

    response = llm(
        full_prompt,
        max_length=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )[0]["generated_text"]

    # Clean output
    answer = response.split("AI:")[-1].strip()

    return answer
