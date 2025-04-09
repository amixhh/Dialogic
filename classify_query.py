import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
embeddings_dir = os.path.join(current_dir, "embeddings")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(os.path.join(embeddings_dir, "faiss_index.index"))
with open(os.path.join(embeddings_dir, "metadata.pkl"), "rb") as f:
    metadata = pickle.load(f)

with open(os.path.join(embeddings_dir, "anchor_embeddings.pkl"), "rb") as f:
    anchor_embeddings = pickle.load(f)
with open(os.path.join(embeddings_dir, "anchor_texts.pkl"), "rb") as f:
    anchor_texts = pickle.load(f)

def classify_with_similarity(query, top_k=1):
    query_embedding = embedder.encode([query])
    similarities = cosine_similarity(query_embedding, anchor_embeddings)[0]
    top_index = np.argmax(similarities)

    best_match = metadata[top_index]
    best_score = similarities[top_index]

    if best_score < 0.4:
        return {"topic": "General", "subtopic": None}

    return best_match

def classify_query(query, threshold=1.0):
    query_vector = embedder.encode([query])
    D, I = index.search(np.array(query_vector), k=1)

    top_match = metadata[I[0][0]]
    top_distance = D[0][0]

    if top_distance <= threshold:
        return top_match

    print("🔁 FAISS fallback: using cosine similarity")
    return classify_with_similarity(query)

if __name__ == "__main__":
    while True:
        user_query = input("Enter your query (or type 'exit'): ")
        if user_query.lower() == 'exit':
            break
        result = classify_query(user_query)
        print("Predicted Topic:", result["topic"])
        print("Predicted Subtopic:", result["subtopic"])
        print("-" * 40)