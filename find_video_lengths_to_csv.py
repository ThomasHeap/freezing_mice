import os
import csv
from pathlib import Path
from moviepy.editor import VideoFileClip

# Define directories for normal and short videos
NORMAL_VIDEO_DIRS = {
    'calms': 'data/calms/videos',
    'foraging': 'data/foraging/videos',
    'freezing': 'data/freezing/videos',
    'grooming': 'data/grooming/videos',
    'mouse_ventral1': 'data/mouse_ventral1/videos',
    'mouse_ventral2': 'data/mouse_ventral2/videos',
    'Scratch-AID': 'data/Scratch-AID/videos',
}
SHORT_VIDEO_DIRS = {
    'calms': 'short_videos/calms',
    'foraging': 'short_videos/foraging',
    'freezing': 'short_videos/freezing',
    'grooming': 'short_videos/grooming',
    'mouse_ventral1': 'short_videos/mouse_ventral1',
    'mouse_ventral2': 'short_videos/mouse_ventral2',
    'Scratch-AID': 'short_videos/Scratch-AID',
}

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}


def find_videos_in_dir(directory):
    """Return a list of video file paths in the given directory."""
    if not os.path.isdir(directory):
        return []
    files = []
    for entry in os.listdir(directory):
        path = os.path.join(directory, entry)
        if os.path.isfile(path) and Path(path).suffix.lower() in VIDEO_EXTENSIONS:
            files.append(path)
    return files


def get_video_length(path):
    try:
        with VideoFileClip(path) as clip:
            return clip.duration
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return None


def main():
    output_csv = 'video_file_lengths.csv'
    rows = []

    # Normal videos
    for category, dir_path in NORMAL_VIDEO_DIRS.items():
        for video_path in find_videos_in_dir(dir_path):
            duration = get_video_length(video_path)
            rows.append({
                'set_type': 'normal',
                'category': category,
                'filename': os.path.basename(video_path),
                'duration_seconds': duration,
                'full_path': os.path.abspath(video_path),
            })

    # Short videos
    for category, dir_path in SHORT_VIDEO_DIRS.items():
        for video_path in find_videos_in_dir(dir_path):
            duration = get_video_length(video_path)
            rows.append({
                'set_type': 'short',
                'category': category,
                'filename': os.path.basename(video_path),
                'duration_seconds': duration,
                'full_path': os.path.abspath(video_path),
            })

    # Write to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['set_type', 'category', 'filename', 'duration_seconds', 'full_path']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote video file lengths to {output_csv}")

if __name__ == '__main__':
    main() 