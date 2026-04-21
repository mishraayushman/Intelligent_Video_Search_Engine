
import faiss
import pickle
import numpy as np
from source.log import logg

def build_faiss(
    embeddings: np.ndarray,
    paths: list[str],
    index_path: str = "index.faiss",
    paths_path: str = "paths.pkl",
    emb_path: str = "embeddings.npy",
) -> faiss.Index:
   
    assert embeddings.dtype == np.float32, "FAISS requires float32"
    assert len(embeddings) == len(paths), "embeddings and paths must have same length"

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    with open(paths_path, "wb") as f:
        pickle.dump(paths, f)

    # Save raw embeddings so search.py can re-rank without index.reconstruct()
    # (reconstruct() is not available on all FAISS index types)
    np.save(emb_path, embeddings)

    logg.info(f"FAISS index built: {len(paths)} vectors, dim={dim}.")
    return index
