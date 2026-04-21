"""
src/embeddings/build_emdd.py
Encodes all frames in a folder into CLIP embeddings.
Deduplicates near-identical frames before storing.
"""

import os
import numpy as np
from source.clip import encode_image
from source.log import logg

def build_embeddings(
    frame_folder: str,
    similarity_threshold: float = 0.85,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode every JPEG in frame_folder into a CLIP embedding.
    Frames that are too similar to an already-stored frame are skipped
    (deduplication), keeping the index compact for CCTV footage.

    Args:
        frame_folder:          Path to folder of extracted JPEG frames.
        similarity_threshold:  Cosine similarity above which a frame is
                               considered a duplicate and skipped (0–1).

    Returns:
        embeddings:  np.ndarray of shape (N, 512), float32, L2-normalised.
        file_paths:  Matching list of frame file paths (len N).
    """
    embeddings: list[np.ndarray] = []
    file_paths: list[str] = []

    frames = sorted(f for f in os.listdir(frame_folder) if f.endswith(".jpg"))
    if not frames:
        raise ValueError(f"No JPEG frames found in '{frame_folder}'")

    for file in frames:
        path = os.path.join(frame_folder, file)

        # encode_image returns shape (1, 512) — flatten to (512,)
        emb = encode_image(path).cpu().numpy().flatten()   # FIX: flatten before dot product

        # Deduplicate: skip if cosine similarity > threshold vs any stored frame
        is_duplicate = False
        for existing in embeddings:
            # Both are 1-D normalised vectors — dot product = cosine similarity
            if float(np.dot(existing, emb)) > similarity_threshold:  # FIX: scalar, not matrix
                is_duplicate = True
                break

        if not is_duplicate:
            embeddings.append(emb)
            file_paths.append(path)

    if not embeddings:
        raise RuntimeError("All frames were deduplicated — nothing to index.")

    logg.info(
        f"Encoded {len(file_paths)}/{len(frames)} frames ",
        f"({len(frames) - len(file_paths)} duplicates removed)."
    )

    return np.vstack(embeddings).astype(np.float32), file_paths
