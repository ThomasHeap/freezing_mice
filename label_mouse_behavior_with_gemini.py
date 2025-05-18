import os
import json
import argparse
from pathlib import Path

from src.models.mouse_behavior import MouseBehaviorSegment
from src.utils.gemini_client import get_gemini_response
from src.scoring.evaluator import score_predictions

def parse_args():
    parser = argparse.ArgumentParser(description='Label mouse behavior using Gemini')
    parser.add_argument('--video', required=True, help='Path to the video file')
    parser.add_argument('--annotation', required=True, help='Path to the annotation JSON file')
    parser.add_argument('--output-dir', default='./gemini_mouse_behavior_results',
                      help='Directory to save results (default: ./gemini_mouse_behavior_results)')
    parser.add_argument('--start-segment', type=int, default=15,
                      help='Number of initial segments to use as context (default: 10)')
    parser.add_argument('--api-key', required=True,
                      help='Google Cloud API key for Gemini')
    parser.add_argument('--model-id', default='gemini-2.0-flash-001',
                      help='Gemini model ID (default: gemini-2.0-flash-001)')
    return parser.parse_args()

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
    
    # Load video
    with open(args.video, "rb") as f:
        video_data = f.read()
    
    # Get Gemini response
    response = get_gemini_response(
        video_data, 
        annotation_summary, 
        args.start_segment,
        args.api_key,
        args.model_id
    )
    
    try:
        model_outputs = response.parsed
        output_path = output_dir / f"{base_name}_gemini_segments.json"
        with open(output_path, "w") as out_f:
            json.dump([seg.model_dump() for seg in model_outputs], out_f, indent=2)
        print(f"Wrote Gemini output to {output_path}")
    except Exception as e:
        print(f"Error parsing Gemini output: {e}")
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
    scores_path = output_dir / f"{base_name}_scores.json"
    with open(scores_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nSaved scores to {scores_path}")

if __name__ == "__main__":
    main() 
    
