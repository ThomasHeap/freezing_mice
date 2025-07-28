#!/usr/bin/env python3
"""
Script to calculate inverse variance weights for datasets based on ground truth annotations.
This script analyzes the variance and entropy of labels in each dataset and calculates
inverse variance weights that can be used for weighted visualization.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math
import os
from sklearn.preprocessing import LabelEncoder
from scipy.stats import entropy

@dataclass
class Segment:
    start_time: float
    end_time: float
    behavior: str

class DatasetWeightCalculator:
    """Calculate inverse variance weights for datasets based on ground truth annotations."""
    
    def __init__(self):
        self.datasets = {}
        self.weights = {}
        
    def time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS format to seconds."""
        if isinstance(time_str, (int, float)):
            return float(time_str)
        if ":" in time_str:
            minutes, seconds = map(float, time_str.split(":"))
            return minutes * 60 + seconds
        else:
            # If no colon, assume it's already in seconds
            return float(time_str)
    
    def load_segments_from_json(self, filepath: str, short_videos: bool, start_segment: int = 0) -> List[Segment]:
        """Load segments from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        segments = []
        if 'segments' in data:
            segment_data = data['segments']
        else:
            segment_data = data
            
        for i, seg in enumerate(segment_data[start_segment:], start=start_segment):
            if short_videos:
                if self.time_to_seconds(seg['start_time']) > 599:
                    break
            start_time = self.time_to_seconds(seg['start_time'])
            end_time = self.time_to_seconds(seg['end_time'])
            behavior = seg['behavior']
            segments.append(Segment(start_time=start_time, end_time=end_time, behavior=behavior))
        
        return segments
    
    def get_behavior_at_second(self, segments: List[Segment], second: int) -> str:
        """Get the behavior occurring at a specific second."""
        for seg in segments:
            # Handle segments with zero duration (start_time == end_time)
            if seg.start_time == seg.end_time:
                if seg.start_time == second:
                    return seg.behavior
            else:
                # Normal case: segment spans multiple seconds
                if seg.start_time <= second <= seg.end_time:
                    return seg.behavior
        return 'background'
    

    
    def calculate_dataset_weights(self, dataset_stats: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate weights for datasets using both entropy and total seconds."""
        weights = {}
        epsilon = 1e-8  # To avoid zero weights

        for dataset_name, stats in dataset_stats.items():
            entropy_val = stats['entropy']
            total_seconds = stats['total_seconds']
            # Combine entropy and total_seconds
            weights[dataset_name] = (entropy_val + epsilon) * total_seconds

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        else:
            num_datasets = len(dataset_stats)
            weights = {name: 1.0 / num_datasets for name in dataset_stats.keys()}

        return weights
    
    def analyze_dataset(self, dataset_path: str, dataset_name: str, short_videos: bool) -> Dict[str, float]:
        """Analyze a single dataset and return its statistics."""
        print(f"Analyzing dataset: {dataset_name}")
        
        # Find all annotation files
        annotation_files = []
        if os.path.isdir(dataset_path):
            for file in os.listdir(dataset_path):
                if file.endswith('.json') and not file.startswith('gt_file_paths'):
                    annotation_files.append(os.path.join(dataset_path, file))
        else:
            annotation_files = [dataset_path]
        
        print(f"  Found {len(annotation_files)} annotation files")
        
        # Process each file individually and collect all per-second labels
        all_labels = []
        total_segments = 0
        
        for file_path in annotation_files:
            try:
                segments = self.load_segments_from_json(file_path, short_videos)
                total_segments += len(segments)
                

                
                # Create per-second labels for this file
                if segments:
                    # Find duration for this file
                    max_time = max(int(seg.end_time) for seg in segments)
                    
                    # Create second-by-second labels for this file
                    file_labels = []
                    for second in range(max_time + 1):
                        label = self.get_behavior_at_second(segments, second)
                        file_labels.append(label)
                    
                    # Add to overall labels
                    all_labels.extend(file_labels)
                
            except Exception as e:
                print(f"  Warning: Could not load {file_path}: {e}")
                continue
        
        print(f"  Total segments loaded: {total_segments}")
        print(f"  Total seconds across all files: {len(all_labels)}")
        
        if not all_labels:
            print(f"  Warning: No labels generated for {dataset_name}")
            return {}
        
        # Calculate statistics on the combined labels
        label_counts = Counter(all_labels)
        total_seconds = len(all_labels)
        
        # Calculate proportions
        proportions = {label: count / total_seconds for label, count in label_counts.items()}
        
        # Calculate variance (using proportions as probabilities)
        mean_prop = np.mean(list(proportions.values()))
        variance = np.var(list(proportions.values()))
        
        # Calculate entropy
        # Convert to probability distribution (normalize)
        prob_dist = np.array(list(proportions.values()))
        prob_dist = prob_dist / np.sum(prob_dist)  # Normalize to sum to 1
        entropy_val = entropy(prob_dist) if np.sum(prob_dist) > 0 else 0
        
        # Calculate Gini coefficient (measure of inequality)
        sorted_props = np.sort(list(proportions.values()))
        n = len(sorted_props)
        gini = 0
        if n > 0:
            for i in range(n):
                gini += (2 * (i + 1) - n - 1) * sorted_props[i]
            gini = gini / (n * np.sum(sorted_props))
        
        stats = {
            'total_seconds': total_seconds,
            'unique_labels': len(label_counts),
            'label_counts': dict(label_counts),
            'proportions': proportions,
            'variance': variance,
            'entropy': entropy_val,
            'gini_coefficient': gini,
            'mean_proportion': mean_prop
        }
        
        print(f"  Total seconds: {stats['total_seconds']}")
        print(f"  Unique labels: {stats['unique_labels']}")
        print(f"  Variance: {stats['variance']:.6f}")
        print(f"  Entropy: {stats['entropy']:.6f}")
        print(f"  Gini coefficient: {stats['gini_coefficient']:.6f}")
        
        return stats
    
    def process_all_datasets(self, data_dir: str, short_videos: bool) -> Dict[str, Dict]:
        """Process all datasets in the data directory."""
        dataset_stats = {}
        
        # Define dataset directories
        dataset_dirs = {
            'calms': 'calms/annotations',
            'grooming': 'grooming/annotations', 
            'freezing': 'freezing/annotations',
            'mouse_ventral1': 'mouse_ventral1/annotations',
            'mouse_ventral2': 'mouse_ventral2/annotations',
            'scratch_aid': 'Scratch-AID/annotations',
            #'foraging': 'foraging/annotations'
        }
        
        for dataset_name, rel_path in dataset_dirs.items():
            dataset_path = os.path.join(data_dir, rel_path)
            if os.path.exists(dataset_path):
                try:
                    stats = self.analyze_dataset(dataset_path, dataset_name, short_videos)
                    dataset_stats[dataset_name] = stats
                except Exception as e:
                    print(f"Error processing {dataset_name}: {e}")
                    continue
            else:
                print(f"Dataset path not found: {dataset_path}")
        
        return dataset_stats
    
    def save_weights(self, weights: Dict[str, float], output_path: str):
        """Save weights to JSON file."""
        output_data = {
            'weights': weights,
            'description': 'Entropy-based weights for datasets based on ground truth label diversity',
            'calculation_method': 'entropy + epsilon normalized to sum to 1'
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nWeights saved to: {output_path}")
        print("\nDataset weights:")
        for dataset, weight in weights.items():
            print(f"  {dataset}: {weight:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Calculate inverse variance weights for datasets')
    parser.add_argument('--data-dir', default='data', 
                       help='Path to data directory containing dataset annotations')
    parser.add_argument('--output', default='dataset_weights.json',
                       help='Output file path for weights')
    parser.add_argument('--short-videos', action='store_true',
                       help='Use short videos instead of full videos')
    args = parser.parse_args()
    
    # Initialize calculator
    calculator = DatasetWeightCalculator()
    
    # Process all datasets
    print("Calculating dataset weights based on ground truth annotations...")
    dataset_stats = calculator.process_all_datasets(args.data_dir, args.short_videos)
    
    if not dataset_stats:
        print("No datasets were successfully processed!")
        return 1
    
    # Calculate weights
    weights = calculator.calculate_dataset_weights(dataset_stats)
    
    # Save weights
    calculator.save_weights(weights, args.output)
    
    # Print summary statistics
    print("\nDataset Statistics Summary:")
    print("-" * 80)
    for dataset_name, stats in dataset_stats.items():
        print(f"\n{dataset_name.upper()}:")
        print(f"  Total seconds: {stats['total_seconds']:,}")
        print(f"  Unique labels: {stats['unique_labels']}")
        print(f"  Variance: {stats['variance']:.6f}")
        print(f"  Entropy: {stats['entropy']:.6f}")
        print(f"  Gini coefficient: {stats['gini_coefficient']:.6f}")
        print(f"  Weight: {weights[dataset_name]:.4f}")
        
        # Print label distribution
        print(f"  Label distribution:")
        for label, count in sorted(stats['label_counts'].items(), key=lambda x: x[1], reverse=True):
            proportion = stats['proportions'][label]
            print(f"    {label}: {count} ({proportion:.3f})")
    
    print(f"\n✅ Weight calculation complete!")
    print(f"📊 Processed {len(dataset_stats)} datasets")
    print(f"💾 Weights saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main()) 