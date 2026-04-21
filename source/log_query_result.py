import csv
import os

def log_results_csv(query: str, results: list, metadata: list):
    
    log_file = "results.csv"
    
    file_exists = os.path.isfile(log_file)  

    time_lookup = {item["frame"]: item["time"] for item in metadata}

    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        if not file_exists:
            writer.writerow(["query_string", "frame_path", "timestamp", "score"])

        for path, score in results:
            timestamp = time_lookup.get(path, 0.0)
            writer.writerow([query, path, timestamp, round(score, 4)])