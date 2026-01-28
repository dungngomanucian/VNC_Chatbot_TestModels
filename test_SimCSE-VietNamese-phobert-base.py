import os
import sys
from dotenv import load_dotenv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from read_test_file import read_test_file

load_dotenv() 
hf_token = os.getenv("HF_TOKEN")

# Chọn file test
test_file = sys.argv[1] if len(sys.argv) > 1 else "test6.txt"
query_chunk, chunks = read_test_file(test_file)

MODEL_NAME = "VoVanPhuc/sup-SimCSE-VietNamese-phobert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def encode_sentences(sentences):
    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]  # CLS pooling
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

print(f"\nquery_chunk: {query_chunk}")

# Embedding cho query_chunk
query_embedding = encode_sentences([query_chunk])

# Embedding cho list chunks
chunk_embeddings = encode_sentences(chunks)

# Tính similarity giữa query_chunk và các chunk trong list
similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

# Tìm top-k chunks gần nhất
k = 5
top_k_indices = np.argsort(similarities)[::-1][:k]
top_k_similarities = similarities[top_k_indices]

print(f"\nTop-{k} chunks gần nhất:")
for i, (idx, sim) in enumerate(zip(top_k_indices, top_k_similarities), 1):
    print(f"Chunk #{idx}: {chunks[idx][:150]}..." if len(chunks[idx]) > 150 else f"Chunk #{idx}: {chunks[idx]}")

print("\nBảng tổng hợp Similarity:")
for i, sim in enumerate(similarities):
    print(f"Chunk #{i}: {sim:.4f}")
