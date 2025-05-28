import os
import json
import argparse
from pathlib import Path
import tempfile
from moviepy.editor import VideoFileClip
import glob 

from src.models.mouse_behavior import MouseBehaviorSegment
from src.utils.gemini_client import get_gemini_response
from src.scoring.evaluator import score_predictions
from src.config.prompts import PROMPTS

def parse_args():
    parser = argparse.ArgumentParser(description='Label mouse behavior using Gemini')
    parser.add_argument('--video', required=True, help='Path to the video file')
    parser.add_argument('--annotation', help='Path to the annotation JSON file')
    parser.add_argument('--output-dir', default='./gemini_mouse_behavior_results',
                      help='Directory to save results (default: ./gemini_mouse_behavior_results)')
    parser.add_argument('--start-segment', type=int, default=0,
                      help='Number of initial segments to use as context (default: 15)')
    parser.add_argument('--max-duration', type=float, default=655,
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
    parser.add_argument('--full-example-annotation',
                      help='Path to a fully annotated example JSON file to use as reference')
    parser.add_argument('--full-example-video',
                      help='Path to the video file corresponding to the fully annotated example')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get base name for output files
    # name after folder name
    base_name = args.video.split("/")[-2] + "_" + args.video.split("/")[-1].split(".")[0]
    
    # Load annotation JSON
    if args.annotation:
        with open(args.annotation, "r") as f:
            annotation_data = json.load(f)
            
        # Prepare annotation summary    
        annotation_summary = json.dumps(annotation_data["segments"][:args.start_segment], indent=2)
    else:
        annotation_data = None
        annotation_summary = None
    if args.full_example_annotation:
        with open(args.full_example_annotation, "r") as f:
            full_example_annotation = json.load(f)
            
        full_example_annotation = json.dumps(full_example_annotation["segments"], indent=2)
    else:
        full_example_annotation = None
    
    example_clips = None
    # if args.prompt_template == "scratching":
    #     example_clips = {'scratching': []}
    #     #load example clips as bytes
    #     for clip in glob.glob(os.path.join('behavior_examples', 'scratching', '*.mp4')):
    #         with open(clip, "rb") as f:
    #             example_clips['scratching'].append(f.read())
    # elif args.prompt_template == "grooming":
    #     example_clips = {'grooming': []}
    #     #load example clips as bytes
    #     for clip in glob.glob(os.path.join('behavior_examples', 'grooming', '*.mp4')):
    #         with open(clip, "rb") as f:
    #             example_clips['grooming'].append(f.read())
    # elif args.prompt_template == "calms":
    #     example_clips = {'attack': [], 'investigation': [], 'mount': []}
    #     #load example clips as bytes
    #     for folder in ['attack', 'investigation', 'mount']:
    #         for clip in glob.glob(os.path.join('behavior_examples', folder, '*.mp4')):
    #             with open(clip, "rb") as f:
    #                 example_clips[folder].append(f.read())
    # else:
    #     example_clips = None
    
    
    
    
    # Get Gemini response
    print("\nCalling Gemini API...")
    
    video_uri = args.video
    full_example_video_uri = args.full_example_video
    
    response = get_gemini_response(
        video_uri, 
        annotation_summary, 
        args.start_segment,
        args.project,
        args.location,
        args.model_id,
        args.prompt_template,
        full_example_annotation,
        full_example_video_uri,
        example_clips
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
    
    # # Score the predictions
    # # Exclude the segments used as context
    # test_annotations = annotation_data["segments"][args.start_segment:]
    # scores = score_predictions(test_annotations, model_outputs)
    
    
    # # Save scores
    # scores_path = output_dir / f"{base_name}_scores_{args.model_id}.json"
    # with open(scores_path, "w") as f:
    #     json.dump(scores, f, indent=2)
    # print(f"\nSaved scores to {scores_path}")

if __name__ == "__main__":
    main() 
    
