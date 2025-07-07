import numpy as np
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
import random
import json
from pathlib import Path

@dataclass
class Segment:
    start_time: float
    end_time: float
    behavior: str

def analyze_segment_durations(gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict[str, Dict[str, float]]:
    """Analyze the distribution of segment durations for each behavior class."""
    gt_durations = defaultdict(list)
    pred_durations = defaultdict(list)
    
    # Calculate durations for ground truth
    for seg in gt_segments:
        duration = seg.end_time - seg.start_time
        gt_durations[seg.behavior].append(duration)
    
    # Calculate durations for predictions
    for seg in pred_segments:
        duration = seg.end_time - seg.start_time
        pred_durations[seg.behavior].append(duration)
    
    # Calculate statistics for each behavior
    stats = {}
    all_behaviors = set(gt_durations.keys()) | set(pred_durations.keys())
    
    for behavior in all_behaviors:
        gt_durs = gt_durations[behavior]
        pred_durs = pred_durations[behavior]
        
        stats[behavior] = {
            "gt_mean": np.mean(gt_durs) if gt_durs else 0,
            "gt_std_error": np.std(gt_durs) / np.sqrt(len(gt_durs)-1) if gt_durs else 0,
            "pred_mean": np.mean(pred_durs) if pred_durs else 0,
            "pred_std_error": np.std(pred_durs) / np.sqrt(len(pred_durs)-1) if pred_durs else 0,
            "gt_count": len(gt_durs),
            "pred_count": len(pred_durs),
            "duration_bias": (np.mean(pred_durs) - np.mean(gt_durs)) if (gt_durs and pred_durs) else 0
        }
    
    return stats

def plot_duration_distributions(gt_segments: List[Segment], pred_segments: List[Segment], 
                              output_dir: Path, behavior: str = None):
    """Plot duration distributions for ground truth and predictions."""
    gt_durations = []
    pred_durations = []
    
    # Collect durations
    for seg in gt_segments:
        if behavior is None or seg.behavior == behavior:
            gt_durations.append(seg.end_time - seg.start_time)
    
    for seg in pred_segments:
        if behavior is None or seg.behavior == behavior:
            pred_durations.append(seg.end_time - seg.start_time)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(gt_durations, bins=20, alpha=0.5, label='Ground Truth', density=True)
    plt.hist(pred_durations, bins=20, alpha=0.5, label='Predictions', density=True)
    
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Density')
    title = f'Duration Distribution{" for " + behavior if behavior else ""}'
    plt.title(title)
    plt.legend()
    
    # Save plot
    filename = f'duration_distribution{"_" + behavior if behavior else ""}.png'
    plt.savefig(output_dir / filename)
    plt.close()

def analyze_temporal_tolerance(gt_segments: List[Segment], pred_segments: List[Segment], 
                             tolerances: List[float] = [0.1, 0.5, 1.0, 2.0]) -> Dict[str, Dict[str, float]]:
    """Analyze boundary detection performance at different temporal tolerances."""
    results = {}
    
    for tolerance in tolerances:
        # Calculate boundary metrics for this tolerance
        gt_boundaries = [(seg.start_time, seg.end_time) for seg in gt_segments]
        pred_boundaries = [(seg.start_time, seg.end_time) for seg in pred_segments]
        
        correct_boundaries = 0
        for gt_start, gt_end in gt_boundaries:
            for pred_start, pred_end in pred_boundaries:
                if (abs(gt_start - pred_start) <= tolerance and 
                    abs(gt_end - pred_end) <= tolerance):
                    correct_boundaries += 1
                    break
        
        precision = correct_boundaries / len(pred_boundaries) if pred_boundaries else 0.0
        recall = correct_boundaries / len(gt_boundaries) if gt_boundaries else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[f"tolerance_{tolerance}"] = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    return results

def generate_confusion_matrix(gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict[str, Dict[str, int]]:
    """Generate a confusion matrix between predicted and ground truth behaviors."""
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # For each ground truth segment, find the best matching prediction
    for gt_seg in gt_segments:
        best_iou = 0
        best_pred = None
        
        for pred_seg in pred_segments:
            # Calculate IoU
            intersection_start = max(gt_seg.start_time, pred_seg.start_time)
            intersection_end = min(gt_seg.end_time, pred_seg.end_time)
            
            if intersection_end <= intersection_start:
                continue
                
            intersection_duration = intersection_end - intersection_start
            union_duration = (gt_seg.end_time - gt_seg.start_time) + \
                           (pred_seg.end_time - pred_seg.start_time) - \
                           intersection_duration
            
            iou = intersection_duration / union_duration if union_duration > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_seg
        
        if best_pred and best_iou > 0.5:  # Only count if IoU > 0.5
            confusion_matrix[gt_seg.behavior][best_pred.behavior] += 1
    
    return dict(confusion_matrix)

def sample_and_visualize_predictions(gt_segments: List[Segment], pred_segments: List[Segment], 
                                   output_dir: Path, num_samples: int = 5):
    """Randomly sample and visualize predictions alongside ground truth."""
    # Get all unique behaviors
    behaviors = set(seg.behavior for seg in gt_segments)
    
    # Sample random segments for each behavior
    for behavior in behaviors:
        gt_samples = [seg for seg in gt_segments if seg.behavior == behavior]
        if not gt_samples:
            continue
            
        samples = random.sample(gt_samples, min(num_samples, len(gt_samples)))
        
        for i, gt_sample in enumerate(samples):
            # Find overlapping predictions
            overlapping_preds = []
            for pred_seg in pred_segments:
                if (pred_seg.start_time <= gt_sample.end_time and 
                    pred_seg.end_time >= gt_sample.start_time):
                    overlapping_preds.append(pred_seg)
            
            # Create visualization
            plt.figure(figsize=(15, 4))
            
            # Plot ground truth
            plt.barh(0, gt_sample.end_time - gt_sample.start_time, 
                    left=gt_sample.start_time, height=0.4, 
                    color='blue', alpha=0.5, label='Ground Truth')
            plt.text(gt_sample.start_time, 0, gt_sample.behavior, 
                    va='center', ha='left')
            
            # Plot predictions
            for j, pred in enumerate(overlapping_preds):
                plt.barh(1, pred.end_time - pred.start_time,
                        left=pred.start_time, height=0.4,
                        color='red', alpha=0.5, label='Prediction' if j == 0 else None)
                plt.text(pred.start_time, 1, pred.behavior,
                        va='center', ha='left')
            
            plt.yticks([0, 1], ['Ground Truth', 'Prediction'])
            plt.xlabel('Time (seconds)')
            plt.title(f'Sample {i+1} - {behavior}')
            plt.legend()
            
            # Save plot
            filename = f'sample_{behavior}_{i+1}.png'
            plt.savefig(output_dir / filename)
            plt.close()

def analyze_all(gt_segments: List[Segment], pred_segments: List[Segment], 
               output_dir: Path) -> Dict:
    """Run all temporal analyses and generate visualizations."""
    results = {}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze segment durations
    duration_stats = analyze_segment_durations(gt_segments, pred_segments)
    results['duration_analysis'] = duration_stats
    
    # Plot duration distributions
    plot_duration_distributions(gt_segments, pred_segments, output_dir)
    for behavior in set(seg.behavior for seg in gt_segments):
        plot_duration_distributions(gt_segments, pred_segments, output_dir, behavior)
    
    # Analyze temporal tolerance
    tolerance_results = analyze_temporal_tolerance(gt_segments, pred_segments)
    results['temporal_tolerance'] = tolerance_results
    
    # Generate confusion matrix
    confusion_matrix = generate_confusion_matrix(gt_segments, pred_segments)
    results['confusion_matrix'] = confusion_matrix
    
    # Sample and visualize predictions
    sample_and_visualize_predictions(gt_segments, pred_segments, output_dir)
    
    # Save results
    with open(output_dir / 'temporal_analysis_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results 