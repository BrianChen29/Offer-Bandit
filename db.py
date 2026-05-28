import os

import faiss
import numpy as np


FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")


def create_faiss_index(embeddings: np.ndarray):
    """Create a FAISS index from a 2D embedding matrix."""
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def save_faiss_index(index, path: str = FAISS_INDEX_PATH) -> None:
    """Save the FAISS index to disk."""
    faiss.write_index(index, path)


def load_faiss_index(path: str = FAISS_INDEX_PATH):
    """Load a FAISS index from disk if it exists."""
    if os.path.exists(path):
        return faiss.read_index(path)
    return None


def get_embeddings_from_vectorstore(vectorstore, query_vector: np.ndarray, k: int = 4):
    """Retrieve the k most similar embeddings for a query vector."""
    return vectorstore.search(query_vector, k)
