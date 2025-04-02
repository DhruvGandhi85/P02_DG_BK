import time
import psutil
import pandas as pd
import src.search
import ollama
from sentence_transformers import SentenceTransformer
import numpy as np
import src.ingest
import sys

def measure_memory():
    """Measure current memory usage in MB."""
    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / (1024 * 1024)


def main(embedding_model, chunk_size, overlap):
    with open("questions.txt", "r") as f:
        questions = [q.strip() for q in f.readlines()]

    results = []

    with open("experiment_results.csv", "a") as f:
        f.write("Model,Question,Response,Speed,Memory (MB),Quality,Chunk Size,Overlay\n")
        for question in questions:
            print(f"Processing question: {question}")
            initial_memory = measure_memory()
            start_time = time.time()
            response = src.search.interactive_search(embedding_model, query=question)
            end_time = time.time()
            final_memory = measure_memory()
            speed = end_time - start_time
            memory_used = final_memory - initial_memory


            quality = "-"
            question = question.replace(",", "")
            results.append([embedding_model, question, response, speed, memory_used, quality, chunk_size, overlap])
            f.write(f"{embedding_model},{question},{response},{speed},{memory_used},{quality},{chunk_size},{overlap}\n")


    print("Experiment complete! Results saved to experiment_results.csv")


if __name__ == "__main__":
    main()