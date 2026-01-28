import os
import sys
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from read_test_file import read_test_file

load_dotenv() 
hf_token = os.getenv("HF_TOKEN")

# Chọn file test
test_file = sys.argv[1] if len(sys.argv) > 1 else "chunks_test/test6.txt"
query_chunk, chunks = read_test_file(test_file)

model = SentenceTransformer("keepitreal/vietnamese-sbert")
model.max_seq_length = 256

print(f"\nquery_chunk: {query_chunk}")

# Embedding cho query_chunk
query_embedding = model.encode(
    query_chunk, 
    convert_to_numpy=True,
    normalize_embeddings=True  
)

# Embedding cho list chunks
chunk_embeddings = model.encode(
    chunks, 
    convert_to_numpy=True,
    normalize_embeddings=True  
)

# Tính similarity giữa query_chunk và các chunk trong list
similarities_sklearn = cosine_similarity(
     query_embedding.reshape(1, -1), 
     chunk_embeddings
)[0]

# Tìm top-k chunks gần nhất
k = 5  # Số lượng chunks gần nhất cần tìm
top_k_indices = np.argsort(similarities_sklearn)[::-1][:k]  
top_k_similarities = similarities_sklearn[top_k_indices]

print(f"\nTop-{k} chunks gần nhất:")

for i, (idx, sim) in enumerate(zip(top_k_indices, top_k_similarities), 1):
    print(f"Chunk #{idx}: {chunks[idx][:150]}..." if len(chunks[idx]) > 150 else f"Chunk #{idx}: {chunks[idx]}")

print("\nBảng tổng hợp Similarity:")
for i, sim in enumerate(similarities_sklearn):
    print(f"Chunk #{i}: {sim:.4f}")