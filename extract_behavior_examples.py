import os
import json
import pandas as pd
from pathlib import Path
import cv2
from moviepy.editor import VideoFileClip
import shutil
from collections import defaultdict

def parse_time_to_seconds(time_str):
    """Convert MM:SS format to seconds."""
    if isinstance(time_str, str):
        minutes, seconds = map(int, time_str.split(':'))
        return minutes * 60 + seconds
    return time_str

def extract_clip(video_path, start_time, end_time, output_path):
    """Extract a clip from a video file."""
    try:
        with VideoFileClip(str(video_path)) as video:
            clip = video.subclip(start_time, end_time)
            clip.write_videofile(str(output_path), codec='libx264', audio=False)
    except Exception as e:
        print(f"Error extracting clip from {video_path}: {e}")

def process_grooming_dataset(data_dir, output_dir):
    """Process grooming dataset annotations."""
    annotations_dir = Path(data_dir) / "grooming" / "annotations"
    videos_dir = Path(data_dir) / "grooming" / "videos"
    
    behavior_examples = defaultdict(list)
    
    for annotation_file in annotations_dir.glob("*_annotations.json"):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        video_path = videos_dir / data["video_file"]
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            continue
            
        for segment in data["segments"]:
            if segment["behavior"] != "other":
                start_time = parse_time_to_seconds(segment["start_time"])
                end_time = parse_time_to_seconds(segment["end_time"])
                
                if end_time - start_time >= 1:  # Only include clips longer than 1 second
                    behavior_examples[segment["behavior"]].append({
                        "video_path": video_path,
                        "start_time": start_time,
                        "end_time": end_time
                    })

    return behavior_examples

def process_scratch_aid_dataset(data_dir, output_dir):
    """Process Scratch-AID dataset annotations."""
    annotation_file = Path(data_dir) / "Scratch-AID" / "annotation" / "Video_annotation_V1-V40.tsv"
    videos_dir = Path(data_dir) / "Scratch-AID" / "videos"
    
    behavior_examples = defaultdict(list)
    
    df = pd.read_csv(annotation_file, sep='\t', header=None, names=['start', 'end', 'video'])
    
    for video_name in df['video'].unique():
        video_path = videos_dir / f"{video_name}.mp4"
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            continue
            
        video_df = df[df['video'] == video_name]
        for _, row in video_df.iterrows():
            start_frame = row['start']
            end_frame = row['end']
            
            # Convert frames to seconds (assuming 30 fps)
            start_time = start_frame / 30
            end_time = end_frame / 30
            
            if end_time - start_time >= 1:  # Only include clips longer than 1 second
                behavior_examples["scratching"].append({
                    "video_path": video_path,
                    "start_time": start_time,
                    "end_time": end_time
                })

    return behavior_examples

def process_calms21_dataset(data_dir, output_dir):
    """Process CALMS21 dataset annotations."""
    annotations_dir = Path(data_dir) / "calms21" / "per_video_annot_segment" / "originals"
    videos_dir = Path(data_dir) / "calms21" / "videos"
    
    behavior_examples = defaultdict(list)
    
    for annotation_file in annotations_dir.glob("*.json"):
        with open(annotation_file, 'r') as f:
            data = json.load(f)
            
        video_name = annotation_file.stem.split('_')[0]
        video_path = videos_dir / f"{video_name}.mp4"
        if not video_path.exists():
            print(f"Video file not found: {video_path}")
            continue
            
        for segment in data["segments"]:
            if segment["behavior"] != "other":
                start_time = segment["start_time"]
                end_time = segment["end_time"]
                
                if end_time - start_time >= 1:  # Only include clips longer than 1 second
                    behavior_examples[segment["behavior"]].append({
                        "video_path": video_path,
                        "start_time": start_time,
                        "end_time": end_time
                    })

    return behavior_examples

def main():
    data_dir = Path("data")
    output_dir = Path("behavior_examples")
    output_dir.mkdir(exist_ok=True)
    
    # Process each dataset
    all_behavior_examples = {}
    
    # Process grooming dataset
    grooming_examples = process_grooming_dataset(data_dir, output_dir)
    all_behavior_examples.update(grooming_examples)
    
    # Process Scratch-AID dataset
    scratch_examples = process_scratch_aid_dataset(data_dir, output_dir)
    all_behavior_examples.update(scratch_examples)
    
    # Process CALMS21 dataset
    calms_examples = process_calms21_dataset(data_dir, output_dir)
    all_behavior_examples.update(calms_examples)
    
    # Extract 5 examples for each behavior
    for behavior, examples in all_behavior_examples.items():
        behavior_dir = output_dir / behavior
        behavior_dir.mkdir(exist_ok=True)
        
        # Sort examples by duration (longest first)
        examples.sort(key=lambda x: x["end_time"] - x["start_time"], reverse=True)
        
        # Take top 5 examples
        for i, example in enumerate(examples[:5]):
            output_path = behavior_dir / f"example_{i+1}.mp4"
            extract_clip(
                example["video_path"],
                example["start_time"],
                example["end_time"],
                output_path
            )
            print(f"Extracted {behavior} example {i+1} to {output_path}")

if __name__ == "__main__":
    main() 