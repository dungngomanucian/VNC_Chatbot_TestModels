import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from read_test_file import read_test_file

load_dotenv() 
hf_token = os.getenv("HF_TOKEN")

# Chọn file test
test_file = sys.argv[1] if len(sys.argv) > 1 else "test6.txt"
query_chunk, chunks = read_test_file(test_file)

model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")
model.max_seq_length = 2048

print(f"\nquery_chunk: {query_chunk}")

# Embedding cho query_chunk
query_embedding = model.encode(query_chunk, convert_to_numpy=True)

# Embedding cho list chunks
chunk_embeddings = model.encode(chunks, convert_to_numpy=True)

# Tính similarity giữa query_chunk và các chunk trong list
similarities = query_embedding @ chunk_embeddings.T

# Tìm top-k chunks gần nhất
k = 4
top_k_indices = np.argsort(similarities)[::-1][:k] 
top_k_similarities = similarities[top_k_indices]

print(f"\nTop-{k} chunks gần nhất:")

for i, (idx, sim) in enumerate(zip(top_k_indices, top_k_similarities), 1):
    print(f"Chunk #{idx}: {chunks[idx][:150]}..." if len(chunks[idx]) > 150 else f"Chunk #{idx}: {chunks[idx]}")

print("\nBảng tổng hợp Similarity:")
for i, sim in enumerate(similarities):
    print(f"Chunk #{i}: {sim:.4f}")