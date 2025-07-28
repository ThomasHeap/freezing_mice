import os
import json
import argparse
from pathlib import Path
import tempfile
from moviepy.editor import VideoFileClip
import glob 
import time
import signal
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Optional


try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not available. GCS directory listing will not work.")

from src.models.mouse_behavior import MouseBehaviorSegment
from src.scoring.evaluator import score_predictions
from src.config.prompts import PROMPTS
from src.utils.qwen_client import get_qwen_response

class TimeoutError(Exception):
    """Custom timeout exception"""
    pass

def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Operation timed out")



def process_single_video(video_path: str, args, output_dir: Path) -> bool:
    """
    Process a single video file with timeout and retry logic
    
    Args:
        video_path: Path to the video file
        args: Command line arguments
        output_dir: Output directory for results
        
    Returns:
        bool: True if successful, False otherwise
    """
    max_retries = args.max_retries
    timeout_seconds = args.timeout
    
    for attempt in range(max_retries):
        try:
            print(f"\nProcessing {video_path} (attempt {attempt + 1}/{max_retries})")
            
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                # Process the video
                result = process_video_file(video_path, args, output_dir)
                signal.alarm(0)  # Cancel timeout
                return result
                
            except TimeoutError:
                signal.alarm(0)  # Cancel timeout
                print(f"Timeout after {timeout_seconds} seconds for {video_path}")
                if attempt < max_retries - 1:
                    print(f"Retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(5)  # Wait before retry
                else:
                    print(f"Failed to process {video_path} after {max_retries} attempts")
                    return False
                    
            except Exception as e:
                signal.alarm(0)  # Cancel timeout
                print(f"Error processing {video_path}: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"Retrying... ({attempt + 1}/{max_retries})")
                    time.sleep(5)  # Wait before retry
                else:
                    print(f"Failed to process {video_path} after {max_retries} attempts")
                    return False
                    
        except Exception as e:
            print(f"Unexpected error processing {video_path}: {str(e)}")
            return False
    
    return False

def process_video_file(video_path: str, args, output_dir: Path) -> bool:
    """
    Process a single video file (extracted from original main function)
    
    Args:
        video_path: Path to the video file
        args: Command line arguments
        output_dir: Output directory for results
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Load annotation JSON if provided
    annotation_data = None
    annotation_summary = None
    if args.annotation:
        with open(args.annotation, "r") as f:
            annotation_data = json.load(f)
        annotation_summary = json.dumps(annotation_data["segments"][:args.start_segment], indent=2)
    
    # Load full example annotation if provided
    full_example_annotation = None
    if args.full_example_annotation:
        with open(args.full_example_annotation, "r") as f:
            full_example_annotation = json.load(f)
        full_example_annotation = json.dumps(full_example_annotation["segments"], indent=2)
    
    # Load example clips
    behaviour_clips_dir = "gs://videos_freezing_mice"
    example_clips = None
    if args.example_clips:
        if args.prompt_template == "scratch_aid":
            example_clips = {'scratch_aid': []}
            for clip in glob.glob(os.path.join(args.example_clips, 'scratching', '*.mp4')):
                example_clips['scratch_aid'].append(f"{behaviour_clips_dir}/{clip}")
        elif args.prompt_template == "grooming":
            example_clips = {'grooming': []}
            for clip in glob.glob(os.path.join(args.example_clips, 'grooming', '*.mp4')):
                example_clips['grooming'].append(f"{behaviour_clips_dir}/{clip}")
        elif args.prompt_template == "calms":
            example_clips = {'attack': [], 'investigation': [], 'mount': []}
            for folder in ['attack', 'investigation', 'mount']:
                for clip in glob.glob(os.path.join(args.example_clips, folder, '*.mp4')):
                    example_clips[folder].append(f"{behaviour_clips_dir}/{clip}")
        elif args.prompt_template == "mouse_box":
            example_clips = {'bedding box': []}
            for clip in glob.glob(os.path.join(args.example_clips, 'bedding box', '*.mp4')):
                example_clips['bedding box'].append(f"{behaviour_clips_dir}/{clip}")
    
    # Get Qwen response
    print(f"Calling Qwen API for {video_path}...")
    
    full_example_video_uri = args.full_example_video
    
    response = get_qwen_response(
        video_path, 
        annotation_summary, 
        args.start_segment,
        args.prompt_template,
        full_example_annotation,
        full_example_video_uri,
        example_clips
    )
    
    if response is None:
        print(f"Error: Failed to get response from Qwen API for {video_path}")
        return False
        
    print(f"Processing Qwen response for {video_path}...")
    
    try:
        # Extract the response content
        response_text = response.choices[0].message.content
        
        # Try to parse the JSON response
        try:
            # Clean up the response text to extract JSON
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            try:
                parsed_response = json.loads(response_text)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response for {video_path}: {str(e)}")
                print(f"Raw response: {response_text}")
                # Write the raw response to a .txt file for inspection
                base_name = f"{Path(video_path).stem}"
                error_path = output_dir / f"{base_name}_{args.model_id}_raw_response.txt"
                with open(error_path, "w") as err_f:
                    err_f.write(response_text)
                print(f"Wrote raw response to {error_path} for manual inspection.")
                return False
            
            if "segments" not in parsed_response:
                print(f"Error: No segments found in response for {video_path}")
                print(f"Response: {response_text}")
                return False
                
            segments = parsed_response["segments"]
            print(f"Successfully processed {len(segments)} segments for {video_path}")
            
            # Generate output filename
            base_name = f"{Path(video_path).stem}"
            output_path = output_dir / f"{base_name}.json"
            
            with open(output_path, "w") as out_f:
                json.dump(segments, out_f, indent=2)
            print(f"Wrote Qwen output to {output_path}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response for {video_path}: {str(e)}")
            print(f"Raw response: {response_text}")
            return False
        
    except Exception as e:
        print(f"Error processing Qwen response for {video_path}: {str(e)}")
        print(f"Raw response: {response}")
        return False

def get_video_files(input_path: str) -> List[str]:
    """
    Get list of video files from input path (file or directory)
    Supports both local paths and Google Cloud Storage paths
    
    Args:
        input_path: Path to file or directory (local or GCS)
        
    Returns:
        List of video file paths
    """
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    # Check if it's a Google Cloud Storage path
    if input_path.startswith('gs://'):
        return get_gcs_video_files(input_path, video_extensions)
    
    # Local file system
    if os.path.isfile(input_path):
        # Single file
        if Path(input_path).suffix.lower() in video_extensions:
            return [input_path]
        else:
            print(f"Warning: {input_path} is not a video file")
            return []
    elif os.path.isdir(input_path):
        # Directory - find all video files
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            video_files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(video_files)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return []

def get_gcs_video_files(gcs_path: str, video_extensions: set) -> List[str]:
    """
    Get video files from Google Cloud Storage path
    
    Args:
        gcs_path: GCS path (gs://bucket/path)
        video_extensions: Set of video file extensions
        
    Returns:
        List of GCS video file paths
    """
    if not GCS_AVAILABLE:
        print("Error: google-cloud-storage library not available")
        return []
    
    # Parse GCS path
    if not gcs_path.startswith('gs://'):
        print(f"Error: {gcs_path} is not a valid GCS path")
        return []
    
    # Remove gs:// prefix
    path_without_prefix = gcs_path[5:]
    
    # Split into bucket and prefix
    if '/' in path_without_prefix:
        bucket_name, prefix = path_without_prefix.split('/', 1)
        # Ensure prefix ends with / for directory listing
        if not prefix.endswith('/'):
            prefix += '/'
    else:
        bucket_name = path_without_prefix
        prefix = ''
    
    try:
        # Initialize GCS client
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List blobs in the bucket with the given prefix
        blobs = bucket.list_blobs(prefix=prefix)
        
        video_files = []
        for blob in blobs:
            # Check if the blob name ends with a video extension
            if any(blob.name.lower().endswith(ext) for ext in video_extensions):
                video_files.append(f"gs://{bucket_name}/{blob.name}")
        
        return sorted(video_files)
        
    except Exception as e:
        print(f"Error listing GCS files: {str(e)}")
        return []

def parse_args():
    parser = argparse.ArgumentParser(description='Label mouse behavior using Qwen')
    parser.add_argument('--input', required=True, 
                      help='Path to video file or directory containing video files')
    parser.add_argument('--annotation', help='Path to the annotation JSON file')
    parser.add_argument('--output-dir', default='./results',
                      help='Directory to save results (default: ./results)')
    parser.add_argument('--start-segment', type=int, default=0,
                      help='Number of initial segments to use as context (default: 0)')
    parser.add_argument('--max-duration', type=float, default=655,
                      help='Maximum duration in seconds to process from the video (default: 655)')
    parser.add_argument('--model-id', default='qwen-vl-max',
                      help='Qwen model ID (default: qwen-vl-max)')
    parser.add_argument('--prompt-template',
                      choices=list(PROMPTS.keys()),
                      help='Prompt template to use (default: default)')
    parser.add_argument('--full-example-annotation',
                      help='Path to a fully annotated example JSON file to use as reference')
    parser.add_argument('--full-example-video',
                      help='Path to the video file corresponding to the fully annotated example')
    parser.add_argument('--example-clips',
                      help='Path to the example clips to use as reference')
    parser.add_argument('--timeout', type=int, default=600,
                      help='Timeout in seconds for each video (default: 600)')
    parser.add_argument('--max-retries', type=int, default=3,
                      help='Maximum number of retries for failed videos (default: 3)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get list of video files to process
    video_files = get_video_files(args.input)
    
    if not video_files:
        print(f"No video files found in {args.input}")
        return
    
    print(f"Found {len(video_files)} video files to process:")
    for video_file in video_files:
        print(f"  - {video_file}")
    
    # Process each video file
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {video_file}")
        print(f"{'='*60}")
        
        start_time = time.time()
        if process_single_video(video_file, args, output_dir):
            successful += 1
            elapsed_time = time.time() - start_time
            print(f"✓ Successfully processed {video_file} in {elapsed_time:.1f}s")
        else:
            failed += 1
            elapsed_time = time.time() - start_time
            print(f"✗ Failed to process {video_file} after {elapsed_time:.1f}s")
    
    # Print summary
    print(f"\n{'='*60}")
    print("PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total videos: {len(video_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(video_files)*100:.1f}%")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 