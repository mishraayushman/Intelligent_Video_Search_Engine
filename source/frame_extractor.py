"""
src/ingestion/frame_extractor.py
Extracts frames from a video at a given FPS and saves metadata.json.
"""

import cv2
import os
import json
from source.log import logg

def extract_frames(video_path: str, output_folder: str, fps: int = 1) -> None:
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval = max(1, int(frame_rate / fps))

    count = 0
    saved = 0
    metadata = []

    while True:                        
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            timestamp = count / frame_rate
            filename = os.path.join(output_folder, f"frame_{saved:06d}.jpg")
            cv2.imwrite(filename, frame)
            metadata.append({"frame": filename, "time": round(timestamp, 3)})
            saved += 1

        count += 1

    cap.release()

    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logg.info(f"Extracted {saved} frames at {fps} fps from '{video_path}'.")
