import faiss
import pickle
import numpy as np

from source.clip import encode_text


def load_index(
    index_path: str = "index.faiss",
    paths_path: str = "paths.pkl",
    emb_path: str = "embeddings.npy",
) -> tuple[faiss.Index, list[str], np.ndarray]:
   
    index = faiss.read_index(index_path)

    with open(paths_path, "rb") as f:
        paths = pickle.load(f)

    embeddings = np.load(emb_path)   # shape (N, 512)

    return index, paths, embeddings


def search(
    index: faiss.Index,
    paths: list[str],
    embeddings: np.ndarray,
    query: str,
    top_k: int = 5,
    time_range: tuple[float, float] | None = None,
    metadata: list[dict] | None = None,
) -> list[tuple[str, float]]:
    
    query_emb = encode_text(query).numpy().flatten()   # shape (512,)

   
    _, indices = index.search(query_emb.reshape(1, -1), top_k * 3)

    # --- Optional temporal filter ---
    if time_range is not None and metadata is not None:
        start_sec, end_sec = time_range
        time_lookup = {item["frame"]: item["time"] for item in metadata}
    else:
        time_lookup = {}

    # --- Stage 2: Re-rank candidates with exact cosine similarity ---
    reranked: list[tuple[str, float]] = []

    for idx in indices[0]:
        if idx < 0:             # FAISS returns -1 for empty slots
            continue

        path = paths[idx]

        # Temporal filter
        if time_lookup:
            frame_time = time_lookup.get(path, 0.0)
            if not (start_sec <= frame_time <= end_sec):
                continue

        # FIX: this block was outside the loop in your original code
        frame_emb = embeddings[idx]                      
        score = float(np.dot(query_emb, frame_emb))      
        reranked.append((path, score))

    # Sort descending by score, return top_k
    reranked.sort(key=lambda x: x[1], reverse=True)
    return reranked[:top_k]
