import os
import json
from pathlib import Path

def time_str_to_seconds(time_str):
    if isinstance(time_str, (int, float)):
        return float(time_str)
    if ":" in time_str:
        minutes, seconds = map(float, time_str.split(":"))
        return minutes * 60 + seconds
    return float(time_str)

def seconds_to_time_str(seconds):
    minutes = int(seconds) // 60
    sec = int(seconds) % 60
    return f"{minutes:02d}:{sec:02d}"

CUTOFF = 599  # 9:59 in seconds

for pred_type in ["pro", "flash"]:
    base_dir = Path(f"results_{pred_type}/zero_shot")
    if not base_dir.exists():
        continue
    for dataset in os.listdir(base_dir):
        dataset_dir = base_dir / dataset / "split_predictions"
        if not dataset_dir.exists():
            continue
        out_dir = base_dir / f"{dataset}_short"
        out_dir.mkdir(parents=True, exist_ok=True)
        for file in dataset_dir.glob("*.json"):
            with open(file) as f:
                segments = json.load(f)
            short_segments = []
            for seg in segments:
                if "start_time" not in seg or "end_time" not in seg:
                    print(f"Warning: Skipping segment in {file} due to missing keys: {seg}")
                    continue
                start = time_str_to_seconds(seg["start_time"])
                end = time_str_to_seconds(seg["end_time"])
                if start >= CUTOFF:
                    break
                seg_copy = dict(seg)
                if end > CUTOFF:
                    seg_copy["end_time"] = seconds_to_time_str(CUTOFF)
                    short_segments.append(seg_copy)
                    break
                short_segments.append(seg_copy)
            with open(out_dir / file.name, "w") as f:
                json.dump(short_segments, f, indent=2) 