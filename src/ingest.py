## DS 4300 Example - from docs

import ollama
import redis
import numpy as np
from redis.commands.search.query import Query
import os
import fitz
import sys
import chromadb
from chromadb.config import Settings

# Vector database selection - 'redis' or 'chroma'
VECTOR_DB = 'chroma'

# Initialize Redis
redis_client = redis.Redis(host="localhost", port=6380, db=0)

# Initialize ChromaDB - could not run on Docker (it doesn't work)
chroma_client = chromadb.Client()
# Get or create a collection
try:
    chroma_collection = chroma_client.get_collection("document_embeddings")
except:
    chroma_collection = chroma_client.create_collection("document_embeddings")

VECTOR_DIM = 768
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# used to clear the vector store
def clear_vector_store():
    if VECTOR_DB == 'redis':
        clear_redis_store()
    elif VECTOR_DB == 'chroma':
        clear_chroma_store()
    else:
        raise ValueError(f"Unknown vector database: {VECTOR_DB}")


# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# used to clear the chroma vector store
def clear_chroma_store():
    print("Clearing existing Chroma store...")
    try:
        global chroma_collection
        chroma_client.delete_collection("document_embeddings")
        chroma_collection = chroma_client.create_collection("document_embeddings")
        print("Chroma store cleared.")
    except Exception as e:
        print(f"Error clearing Chroma store: {e}")


# An HNSW index in Redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# store the embedding in the selected vector store
def store_embedding(file: str, page: str, chunk: str, text: str, embedding: list):
    if VECTOR_DB == 'redis':
        store_embedding_redis(file, page, chunk, embedding)
    elif VECTOR_DB == 'chroma':
        store_embedding_chroma(file, page, chunk, text, embedding)
    else:
        raise ValueError(f"Unknown vector database: {VECTOR_DB}")


# store the embedding in Redis
def store_embedding_redis(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding in Redis for: {chunk[:30]}...")


# store the embedding in Chroma
def store_embedding_chroma(file: str, page: str, chunk: str, text: str, embedding: list):
    chroma_collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
        ids=[f"{file}_page_{page}_chunk_{chunk[:20]}"]  # Using first 20 chars of chunk as part of ID
    )
    print(f"Stored embedding in Chroma for: {chunk[:30]}...")


# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=400, overlap=100):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# process all PDF files in a directory
def process_pdfs(model, data_dir, chunk_size=400, overlap=100):
    print(f"Processing PDFs using {VECTOR_DB} as vector store...")
    
    # Initialize vector store
    if VECTOR_DB == 'redis':
        create_hnsw_index()

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text, chunk_size, overlap)
                for chunk_index, chunk in enumerate(chunks):
                    embedding = get_embedding(chunk, model=model)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk=str(chunk_index),
                        text=chunk,
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")


def query_redis(model, query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    embedding = get_embedding(query_text, model=model)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


def main():
    clear_vector_store()
    
    # data directory and chunk parameters (because of problems with docker)
    data_dir = "./data/"
    chunk_size = 400
    overlap = 100
    
    process_pdfs(MODEL, data_dir, chunk_size, overlap)
    print("\n---Done processing PDFs---\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODEL = sys.argv[1]
    else:
        MODEL = "deepseek-r1"
    main()
    