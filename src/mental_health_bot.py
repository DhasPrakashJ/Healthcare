import pandas as pd
from rag_utils import build_faiss_index, retrieve_similar, generate_response


# ---------------------------
# Load Knowledge Base
# ---------------------------
def load_knowledge_base(path: str):
    try:
        kb = pd.read_csv(path)
        return kb["content"].dropna().tolist()
    except Exception as e:
        print(f"Error loading knowledge base: {e}")
        return []


# ---------------------------
# Initialize RAG System
# ---------------------------
def initialize_rag(texts):
    if not texts:
        raise ValueError("Knowledge base is empty.")
    index, embeddings = build_faiss_index(texts)
    return index, embeddings


# ---------------------------
# Generate AI Response
# ---------------------------
def get_ai_response(user_input, texts, index, embeddings):
    retrieved_docs = retrieve_similar(
        user_input, texts, index, embeddings, top_k=3
    )
    context = "\n".join([doc for doc, _ in retrieved_docs])

    response = generate_response(user_input, context=context)
    return response


# ---------------------------
# Chat Interface
# ---------------------------
def chat():
    print("\n===== AI Mental Health Companion =====")
    print("Note: This is a supportive AI, not a licensed therapist.\n")

    name = input("Your Name: ")
    print(f"\nHello {name}, I'm here to listen and support you.\n")

    # Load + Initialize
    texts = load_knowledge_base("data/knowledge_base/articles.csv")
    index, embeddings = initialize_rag(texts)

    while True:
        user_input = input("You: ")

        if user_input.lower().strip() in ["exit", "quit"]:
            print("\nTake care. Please seek professional help if needed.")
            break

        try:
            response = get_ai_response(user_input, texts, index, embeddings)
            print(f"\nAI: {response}\n")
        except Exception as e:
            print(f"\nError generating response: {e}\n")


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    chat()
