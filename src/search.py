import redis
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField
import sys
import chromadb
from chromadb.config import Settings

# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Vector database selection - 'redis' or 'chroma'
VECTOR_DB = 'chroma'

# Redis configuration
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

# ChromaDB configuration
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
MODEL = '' # select model installed in ollama, check model by running 'ollama list' in terminal
# tested deepseek-r1, mistral, llama, and a few others


# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_embeddings(model, query, top_k=3):
    query_embedding = get_embedding(query, model=model)
    
    if VECTOR_DB == 'redis':
        return search_embeddings_redis(query_embedding, top_k)
    elif VECTOR_DB == 'chroma':
        return search_embeddings_chroma(query, query_embedding, top_k)
    else:
        raise ValueError(f"Unknown vector database: {VECTOR_DB}")


def search_embeddings_redis(query_embedding, top_k=3):
    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Redis search error: {e}")
        return []


def search_embeddings_chroma(query_text, query_embedding, top_k=3):
    try:
        # Query the collection
        results = chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        # Transform results into the expected format
        top_results = []
        
        if results and 'metadatas' in results and results['metadatas']:
            for i, metadata in enumerate(results['metadatas'][0]):
                similarity = 1.0 - float(results['distances'][0][i]) if 'distances' in results else 0.0
                top_results.append({
                    "file": metadata.get('file', 'Unknown'),
                    "page": metadata.get('page', 'Unknown'),
                    "chunk": metadata.get('chunk', 'Unknown'),
                    "similarity": similarity,
                })
        
        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )
            
        return top_results
        
    except Exception as e:
        print(f"Chroma search error: {e}")
        return []


def generate_rag_response(embedding_model, query, context_results):
    # Prepare context string with better formatting and relevance indicators
    context_items = []
    for i, result in enumerate(context_results):
        similarity = float(result.get('similarity', 0))
        # Format with clear separation between sources
        context_items.append(
            f"[Source {i+1}] From file: {result.get('file', 'Unknown file')}\n"
            f"Page: {result.get('page', 'Unknown page')}, Section: {result.get('chunk', 'Unknown chunk')}\n"
            f"Relevance score: {similarity:.2f}\n"
            f"Content: {result.get('text', '')}\n"
        )
    
    context_str = "\n".join(context_items)

    print(f"Retrieved {len(context_results)} relevant documents")

    # Improved prompt with better instructions
    prompt = f"""You are a knowledgeable assistant answering questions based on specific document sources.

            CONTEXT:
            {context_str}

            USER QUESTION: {query}

            INSTRUCTIONS:
            1. Answer ONLY based on the information provided in the context.
            2. If the context doesn't contain relevant information, respond with "I don't have enough information to answer this question."
            3. Do not make up or infer information that isn't explicitly stated in the context.
            4. Cite the specific sources you used (e.g., [Source 1], [Source 2]) when providing your answer.
            5. Be concise but thorough.

            YOUR ANSWER:"""

    # Generate response using Ollama
    try:
        response = ollama.chat(
            model=embedding_model, 
            messages=[{"role": "user", "content": prompt}], 
            options={"num_predict": 512}  # higher token limit
        )
        return response["message"]["content"]
    except Exception as e:
        print(f"Error generating response: {e}")
        fallback_model = "mistral"
        try:
            print(f"Falling back to {fallback_model}")
            response = ollama.chat(
                model=fallback_model, 
                messages=[{"role": "user", "content": prompt}], 
                options={"num_predict": 256}
            )
            return response["message"]["content"]
        except:
            return "I'm sorry, I couldn't generate a response due to a technical issue."


def interactive_search(model, query=None):
    """Interactive search interface."""
    print(f"🔍 RAG Search Interface (Using {VECTOR_DB} database)")
    print("Type 'exit' to quit")

    while True:
        if query is None:
            query = input("\nEnter your search query: ")
        else:
            query = query.strip()
        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(model, query)

        # Generate RAG response
        response = generate_rag_response(model, query, context_results)
        response = response.replace(",", "")
        response = response.replace("\n", "  ")

        print("\n--- Response ---")
        print(response)
        break

    return response


# Store embedding functions for both Redis and Chroma
def store_embedding_redis(file, page, chunk, embedding):
    """Store an embedding in Redis using a hash with vector field."""
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
        },
    )


def store_embedding_chroma(file, page, chunk, text, embedding):
    """Store an embedding in ChromaDB."""
    chroma_collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[{"file": file, "page": page, "chunk": chunk}],
        ids=[f"{file}_page_{page}_chunk_{chunk}"]
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODEL = sys.argv[1]
    else:
        MODEL = "llama3.2"
    interactive_search(MODEL)