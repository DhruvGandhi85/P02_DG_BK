# Ollama RAG Ingest and Search

## Prerequisites

- Ollama app set up ([Ollama.com](Ollama.com))
- Python with Ollama, Redis-py, and Numpy installed (`pip install ollama redis numpy`)
- Redis Stack running (Docker container is fine) on port 6379.  If that port is mapped to another port in 
Docker, change the port number in the creation of the Redis client in both python files in `src`.

## Source Code
- `src/ingest.py` - imports and processes PDF files in `./data` folder. Embeddings and associated information 
stored in Redis-stack
- `src/search.py` - simple question answering using 

# Run Code
- Set up ollama environment and perform `ollama run {model}` within terminal window.
- Ensure Redis Stack running on port 6380 or ChromaDB running.
- ingest pdfs using `src/ingest.py`. Insert `{model}` as parameter.
- run search retrieval with `src/search.py`. Insert `{model}` as parameter.
