import os
import sys
from dotenv import load_dotenv
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from read_test_file import read_test_file

load_dotenv() 
hf_token = os.getenv("HF_TOKEN")

# Chọn file test 
test_file = sys.argv[1] if len(sys.argv) > 1 else "chunks_test/test6.txt"
query_chunk, chunks = read_test_file(test_file)

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base')
model = AutoModel.from_pretrained('intfloat/multilingual-e5-base')

print(f"\nquery_chunk: {query_chunk}")

# Embedding cho query chunk
query_text = f"query: {query_chunk}"
batch_dict_query = tokenizer([query_text], max_length=512, padding=True, truncation=True, return_tensors='pt')
outputs_query = model(**batch_dict_query)
query_embedding = average_pool(outputs_query.last_hidden_state, batch_dict_query['attention_mask'])
query_embedding = F.normalize(query_embedding, p=2, dim=1)
query_embedding = query_embedding.detach().numpy()

# Embedding cho list chunks
chunk_texts = [f"passage: {chunk}" for chunk in chunks]
batch_dict_chunks = tokenizer(chunk_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
outputs_chunks = model(**batch_dict_chunks)
chunk_embeddings = average_pool(outputs_chunks.last_hidden_state, batch_dict_chunks['attention_mask'])
chunk_embeddings = F.normalize(chunk_embeddings, p=2, dim=1)
chunk_embeddings = chunk_embeddings.detach().numpy()

# Tính similarity giữa query_chunk và các chunk trong list
similarities = (query_embedding @ chunk_embeddings.T)[0]  # Lấy phần tử đầu tiên vì query_embedding là 2D

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
