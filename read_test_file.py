import re

def read_test_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Tìm query_chunk
    query_match = re.search(r'query_chunk\s*=\s*"([^"]+)"', content)
    if not query_match:
        raise ValueError(f"Không tìm thấy query_chunk trong file {file_path}")
    query_chunk = query_match.group(1)
    
    # Tìm chunks
    chunks_match = re.search(r'chunks\s*=\s*\[(.*?)\]', content, re.DOTALL)
    if not chunks_match:
        raise ValueError(f"Không tìm thấy chunks trong file {file_path}")
    
    chunks_content = chunks_match.group(1)
    # Tách các chunk bằng regex
    chunk_pattern = r'"([^"]+)"'
    chunks = re.findall(chunk_pattern, chunks_content)
    
    return query_chunk, chunks