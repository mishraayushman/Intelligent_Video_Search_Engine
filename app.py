

import streamlit as st
import json
import os          
import shutil

from source.frame_extractor import extract_frames
from source.build_emdd import build_embeddings
from source.faiss1 import build_faiss
from source.search import search, load_index
from source.log_query_result import log_results_csv

st.set_page_config(page_title="Video Search", page_icon="🎥", layout="wide")
st.title("🎥 Video Semantic Search Engine")

uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    video_path = "uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("Video uploaded!")

    if st.button("Process Video"):

        if os.path.exists("frames"):
            shutil.rmtree("frames")
        os.makedirs("frames", exist_ok=True)

        with st.spinner("Extracting frames..."):
            extract_frames(video_path, "frames", fps=1)

        with st.spinner("Generating embeddings..."):
            embeddings, frame_paths = build_embeddings("frames")

        with st.spinner("Building FAISS index..."):
            build_faiss(embeddings, frame_paths)

        st.success("Processing complete! You can now search.")


        index, paths, embs = load_index()          
        with open("metadata.json", "r") as f:
            metadata = json.load(f)

        try:
            with open("ground_truth.json", "r") as f:
                gt_list = json.load(f)
                gt_dict = {item["query"].lower(): item for item in gt_list}
        except FileNotFoundError:
            gt_dict = {}
        # ==========================================

        st.session_state["index"]     = index
        st.session_state["paths"]     = paths
        st.session_state["embs"]      = embs
        st.session_state["metadata"]  = metadata
        st.session_state["ground_truth"] = gt_dict


if "index" in st.session_state:
    st.divider()
    col1, col2 = st.columns([3, 1])

    with col1:
        query = st.text_input("🔍 Search query", placeholder='e.g. "person carrying a bag near entrance"')

    with col2:
        top_k = st.slider("Results", min_value=1, max_value=10, value=5)

    # Optional temporal filter
    use_time_filter = st.checkbox("Filter by time range")
    time_range = None
    if use_time_filter:
        metadata = st.session_state["metadata"]
        max_time = max(item["time"] for item in metadata)
        t_start, t_end = st.slider(
            "Time range (seconds)",
            min_value=0.0, max_value=float(max_time),
            value=(0.0, float(max_time)), step=1.0
        )
        time_range = (t_start, t_end)

    if query:
        results = search(
            index=st.session_state["index"],
            paths=st.session_state["paths"],
            embeddings=st.session_state["embs"],
            query=query,
            top_k=top_k,
            time_range=time_range,
            metadata=st.session_state.get("metadata"),
        )

        if not results:
            st.warning("No results found. Try a different query or widen the time range.")
        else:
            log_results_csv(query, results, st.session_state["metadata"])
            st.subheader(f"Top {len(results)} results for: *{query}*")
            metadata = st.session_state["metadata"]
            time_lookup = {item["frame"]: item["time"] for item in metadata}

            gt_dict = st.session_state.get("ground_truth", {})
            query_lower = query.strip().lower()

            # Only evaluate if this exact query exists in our ground truth file
            if query_lower in gt_dict:
                expected_start = gt_dict[query_lower]["expected_start"]
                expected_end = gt_dict[query_lower]["expected_end"]
                
                correct_hits = 0
                first_correct_rank = None
                
                # Check each returned frame against the answer key
                for rank, (path, score) in enumerate(results, start=1):
                    timestamp = time_lookup.get(path, 0.0)
                    if expected_start <= timestamp <= expected_end:
                        correct_hits += 1
                        if first_correct_rank is None:
                            first_correct_rank = rank
                
                # Calculate metrics
                precision = correct_hits / len(results)
                mrr = 1.0 / first_correct_rank if first_correct_rank else 0.0
                
                # Display metrics in a clean UI box
                st.info("📊 **Live Evaluation: Ground Truth Match Found!**")
                met_col1, met_col2, met_col3 = st.columns(3)
                met_col1.metric(label=f"Precision@{top_k}", value=f"{precision:.2f}")
                met_col2.metric(label="MRR", value=f"{mrr:.2f}")
                if first_correct_rank:
                    met_col3.metric(label="First Correct Result", value=f"Rank #{first_correct_rank}")
                else:
                    met_col3.metric(label="First Correct Result", value="None Found")
                
                st.divider()

            cols = st.columns(min(len(results), 3))
            for i, (path, score) in enumerate(results):
                with cols[i % 3]:
                    timestamp = time_lookup.get(path, 0.0)
                    h = int(timestamp // 3600)
                    m = int((timestamp % 3600) // 60)
                    s = int(timestamp % 60)
                    ts_str = f"{h:02d}:{m:02d}:{s:02d}"
                    st.image(path, use_container_width=True)
                    st.caption(f"{ts_str}  |  Score: {score:.4f}")
