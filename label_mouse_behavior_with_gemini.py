import os
import json
import argparse
from pathlib import Path
import tempfile
from moviepy.editor import VideoFileClip

from src.models.mouse_behavior import MouseBehaviorSegment
from src.utils.gemini_client import get_gemini_response
from src.scoring.evaluator import score_predictions
from src.config.prompts import PROMPTS

def parse_args():
    parser = argparse.ArgumentParser(description='Label mouse behavior using Gemini')
    parser.add_argument('--video', required=True, help='Path to the video file')
    parser.add_argument('--annotation', required=True, help='Path to the annotation JSON file')
    parser.add_argument('--output-dir', default='./gemini_mouse_behavior_results',
                      help='Directory to save results (default: ./gemini_mouse_behavior_results)')
    parser.add_argument('--start-segment', type=int, default=0,
                      help='Number of initial segments to use as context (default: 15)')
    parser.add_argument('--max-duration', type=float, default=200,
                      help='Maximum duration in seconds to process from the video (default: None)')
    # parser.add_argument('--api-key', required=True,
    #                   help='Google Cloud API key for Gemini')
    parser.add_argument('--project',
                      help='Google Cloud project')
    parser.add_argument('--location', default='us-central1',
                      help='Google Cloud location (default: us-central1)')
    parser.add_argument('--model-id', default='gemini-2.0-flash-001',
                      help='Gemini model ID (default: gemini-2.0-flash-001)')
    parser.add_argument('--prompt-template',
                      choices=list(PROMPTS.keys()),
                      help='Prompt template to use (default: default)')
    parser.add_argument('--full-example',
                      help='Path to a fully annotated example JSON file to use as reference')
    parser.add_argument('--full-example-video',
                      help='Path to the video file corresponding to the fully annotated example')
    return parser.parse_args()

# def parse_model_outputs(text):
#     """Parse model outputs from text up to the last valid segment.
    
#     Args:
#         text (str): Raw text output from the model
        
#     Returns:
#         list[MouseBehaviorSegment]: List of valid segments
#     """
#     try:
#         # Find the last valid JSON object in the text
#         segments = []
#         current_segment = {}
#         in_segment = False
        
#         for line in text.split('\n'):
#             line = line.strip()
            
#             # Start of a new segment
#             if line.startswith('{'):
#                 in_segment = True
#                 current_segment = {}
#                 continue
                
#             # End of a segment
#             if line.startswith('}'):
#                 in_segment = False
#                 try:
#                     # Try to create a MouseBehaviorSegment from the current segment
#                     segment = MouseBehaviorSegment(**current_segment)
#                     segments.append(segment)
#                 except Exception:
#                     # If validation fails, stop parsing
#                     break
#                 continue
                
#             # Parse segment fields
#             if in_segment and ':' in line:
#                 key, value = line.split(':', 1)
#                 key = key.strip().strip('"')
#                 value = value.strip().strip(',').strip('"')
                
#                 # Convert numeric values
#                 if key in ['segment_number', 'start_time', 'end_time']:
#                     try:
#                         value = float(value)
#                         if key == 'segment_number':
#                             value = int(value)
#                     except ValueError:
#                         continue
                        
#                 current_segment[key] = value
                
#         return segments
#     except Exception as e:
#         print(f"Error parsing model outputs: {str(e)}")
#         return []

def filter_segments_by_duration(segments, max_duration):
    """Filter segments to only include those within the specified duration.
    
    Args:
        segments (list): List of segment dictionaries
        max_duration (float): Maximum duration in seconds
        
    Returns:
        list: Filtered list of segments
    """
    if max_duration is None:
        return segments
        
    return [seg for seg in segments if seg.get('start_time', 0) < max_duration]

def trim_video(video_path, max_duration):
    """Trim video to specified duration.
    
    Args:
        video_path (str): Path to input video
        max_duration (float): Maximum duration in seconds
        
    Returns:
        bytes: Trimmed video data
    """
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
        temp_path = temp_file.name
        
    try:
        # Load video and trim it
        video = VideoFileClip(video_path)
        trimmed_video = video.subclip(0, min(max_duration, video.duration))
        trimmed_video.write_videofile(temp_path, codec='libx264')
        
        # Read the trimmed video data
        with open(temp_path, 'rb') as f:
            video_data = f.read()
            
        return video_data
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base name for output files
    base_name = Path(args.video).stem
    
    # Load annotation JSON
    with open(args.annotation, "r") as f:
        annotation_data = json.load(f)
    
    # Prepare annotation summary
    annotation_summary = json.dumps(annotation_data["segments"][:args.start_segment], indent=2)
    
    # Load full example if provided
    full_example = None
    full_example_video = None
    if args.full_example:
        with open(args.full_example, "r") as f:
            full_example = json.load(f)
            if args.max_duration:
                # Only apply max duration to example segments
                full_example["segments"] = filter_segments_by_duration(full_example["segments"], args.max_duration)
                print(f"\nFiltered example to {len(full_example['segments'])} segments within {args.max_duration} seconds")
        if args.full_example_video:
            if args.max_duration:
                # Only trim example video
                full_example_video = trim_video(args.full_example_video, args.max_duration)
            else:
                with open(args.full_example_video, "rb") as f:
                    full_example_video = f.read()
    
    # Load main video (no trimming)
    with open(args.video, "rb") as f:
        video_data = f.read()
    
    # Get Gemini response
    print("\nCalling Gemini API...")
    
    response = get_gemini_response(
        video_data, 
        annotation_summary, 
        args.start_segment,
        args.project,
        args.location,
        args.model_id,
        args.prompt_template,
        full_example,
        full_example_video
    )
    
    if response is None:
        print("Error: Failed to get response from Gemini API")
        return
        
    print("\nProcessing Gemini response...")
    try:
        
        # Try to get parsed output first
        model_outputs = response.parsed
        
        
        if not model_outputs:
            print("Error: No valid segments found in response")
            print(response.text)
            print("Number of Tokens: ", response.usage_metadata.total_token_count)
            return
            #trying to parse up to last valid segment from the text
            #model_outputs = parse_model_outputs(response.text)
            
        print(f"Successfully processed {len(model_outputs)} segments")
        
        output_path = output_dir / f"{base_name}_gemini_segments_{args.model_id}.json"
        with open(output_path, "w") as out_f:
            json.dump([seg.model_dump() for seg in model_outputs], out_f, indent=2)
        print(f"Wrote Gemini output to {output_path}")
    except Exception as e:
        print(f"Error processing Gemini response: {str(e)}")
        print(f"Raw response: {response}")
        return
    
    # Score the predictions
    # Exclude the segments used as context
    test_annotations = annotation_data["segments"][args.start_segment:]
    scores = score_predictions(test_annotations, model_outputs)
    
    # Print results
    print("\nScoring Results:")
    print(f"Precision: {scores['precision']:.3f}")
    print(f"Recall: {scores['recall']:.3f}")
    print(f"F1 Score: {scores['f1_score']:.3f}")
    print(f"\nDetailed Metrics:")
    print(f"True Positives: {scores['true_positives']}")
    print(f"False Positives: {scores['false_positives']}")
    print(f"False Negatives: {scores['false_negatives']}")
    
    # Save scores
    scores_path = output_dir / f"{base_name}_scores_{args.model_id}.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nSaved scores to {scores_path}")

if __name__ == "__main__":
    main() 
    
