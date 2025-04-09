import json
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

with open("data/dataset.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

texts = [item["text"] for item in data]
metadata = [{"topic": item["topic"], "subtopic": item["subtopic"]} for item in data]

model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

Path("embeddings").mkdir(parents=True, exist_ok=True)
faiss.write_index(index, "embeddings/faiss_index.index")

with open("embeddings/anchor_texts.pkl", "wb") as f:
    pickle.dump(texts, f)

with open("embeddings/anchor_embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)