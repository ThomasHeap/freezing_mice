import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Segment:
    start_time: float
    end_time: float
    behavior: str

class ActionSegmentationEvaluator:
    """Comprehensive evaluation for action segmentation systems at per-second granularity."""
    
    def __init__(self, tolerance_seconds: float = 1.0):
        self.tolerance = tolerance_seconds
        
    def time_to_seconds(self, time_str: str) -> float:
        """Convert MM:SS format to seconds."""
        if isinstance(time_str, (int, float)):
            return float(time_str)
        minutes, seconds = map(float, time_str.split(":"))
        return minutes * 60 + seconds
    
    def calculate_iou(self, seg1: Segment, seg2: Segment) -> float:
        """Calculate Intersection over Union between two segments."""
        intersection_start = max(seg1.start_time, seg2.start_time)
        intersection_end = min(seg1.end_time, seg2.end_time)
        
        if intersection_end <= intersection_start:
            return 0.0
        
        intersection_duration = intersection_end - intersection_start
        union_duration = (seg1.end_time - seg1.start_time) + (seg2.end_time - seg2.start_time) - intersection_duration
        
        return intersection_duration / union_duration if union_duration > 0 else 0.0
    
    def get_behavior_at_second(self, segments: List[Segment], second: int, window_size: int = 0) -> str:
        """Get the behavior occurring at a specific second."""
        # Check for exact matches first
        for seg in segments:
            if seg.start_time <= second < seg.end_time:
                return seg.behavior
        
        # If window_size > 0, check within tolerance window
        if window_size > 0:
            for seg in segments:
                if (seg.start_time - window_size <= second <= seg.end_time + window_size):
                    return seg.behavior
        
        return 'background'  # Default background class
    
    def second_wise_accuracy(self, gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict[str, float]:
        """Calculate second-wise accuracy with detailed per-class metrics."""
        # Find total duration
        max_time = max(max(int(seg.end_time) for seg in gt_segments), 
                      max(int(seg.end_time) for seg in pred_segments))
        
        # Create second-by-second labels
        gt_labels = []
        pred_labels = []
        
        for second in range(max_time + 1):
            gt_label = self.get_behavior_at_second(gt_segments, second)
            pred_label = self.get_behavior_at_second(pred_segments, second)
            gt_labels.append(gt_label)
            pred_labels.append(pred_label)
        
        # Calculate overall accuracy
        correct = sum(1 for gt, pred in zip(gt_labels, pred_labels) if gt == pred)
        total = len(gt_labels)
        
        # Calculate per-class metrics
        class_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0})
        all_classes = set(gt_labels + pred_labels) - {'background'}
        
        for gt, pred in zip(gt_labels, pred_labels):
            for cls in all_classes:
                if gt == cls and pred == cls:
                    class_metrics[cls]['tp'] += 1
                elif gt != cls and pred == cls:
                    class_metrics[cls]['fp'] += 1
                elif gt == cls and pred != cls:
                    class_metrics[cls]['fn'] += 1
                else:
                    class_metrics[cls]['tn'] += 1
        
        # Calculate per-class precision, recall, f1
        per_class_results = {}
        for cls in all_classes:
            tp = class_metrics[cls]['tp']
            fp = class_metrics[cls]['fp']
            fn = class_metrics[cls]['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_results[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'support': tp + fn  # Number of true instances
            }
        
        return {
            'second_accuracy': correct / total,
            'per_class_metrics': per_class_results,
            'total_seconds': total
        }
    
    def segment_iou_metrics(self, gt_segments: List[Segment], pred_segments: List[Segment], 
                           iou_thresholds: List[float] = [0.3, 0.5, 0.7]) -> Dict[str, float]:
        """Calculate segment-level IoU metrics at multiple thresholds."""
        results = {}
        
        for threshold in iou_thresholds:
            tp = 0
            fp = len(pred_segments)
            fn = len(gt_segments)
            
            matched_gt = set()
            matched_pred = set()
            
            # Find matches
            for i, gt_seg in enumerate(gt_segments):
                for j, pred_seg in enumerate(pred_segments):
                    if (gt_seg.behavior == pred_seg.behavior and 
                        j not in matched_pred and 
                        i not in matched_gt):
                        
                        iou = self.calculate_iou(gt_seg, pred_seg)
                        if iou >= threshold:
                            tp += 1
                            matched_gt.add(i)
                            matched_pred.add(j)
                            break
            
            fp = len(pred_segments) - len(matched_pred)
            fn = len(gt_segments) - len(matched_gt)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[f'iou_{threshold}_precision'] = precision
            results[f'iou_{threshold}_recall'] = recall
            results[f'iou_{threshold}_f1'] = f1
        
        # Calculate mean Average Precision (mAP)
        ap_scores = []
        for threshold in np.arange(0.1, 1.0, 0.1):
            tp = 0
            matched_gt = set()
            matched_pred = set()
            
            for i, gt_seg in enumerate(gt_segments):
                for j, pred_seg in enumerate(pred_segments):
                    if (gt_seg.behavior == pred_seg.behavior and 
                        j not in matched_pred and 
                        i not in matched_gt):
                        
                        iou = self.calculate_iou(gt_seg, pred_seg)
                        if iou >= threshold:
                            tp += 1
                            matched_gt.add(i)
                            matched_pred.add(j)
                            break
            
            fp = len(pred_segments) - len(matched_pred)
            fn = len(gt_segments) - len(matched_gt)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            ap_scores.append(precision)
        
        results['mAP'] = np.mean(ap_scores)
        
        return results
    
    def edit_distance(self, gt_segments: List[Segment], pred_segments: List[Segment]) -> int:
        """Calculate edit distance between action sequences."""
        gt_sequence = [seg.behavior for seg in gt_segments]
        pred_sequence = [seg.behavior for seg in pred_segments]
        
        m, n = len(gt_sequence), len(pred_sequence)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gt_sequence[i-1] == pred_sequence[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = min(
                        dp[i-1][j-1] + 1,  # substitution
                        dp[i-1][j] + 1,    # deletion
                        dp[i][j-1] + 1     # insertion
                    )
        
        return dp[m][n]
    
    def boundary_detection_metrics(self, gt_segments: List[Segment], pred_segments: List[Segment], 
                                 tolerance: float = None) -> Dict[str, float]:
        """Evaluate boundary detection accuracy at second-level precision."""
        if tolerance is None:
            tolerance = self.tolerance
        
        # Extract boundaries (start and end points), rounded to nearest second
        gt_boundaries = []
        for seg in gt_segments:
            gt_boundaries.extend([int(round(seg.start_time)), int(round(seg.end_time))])
        
        pred_boundaries = []
        for seg in pred_segments:
            pred_boundaries.extend([int(round(seg.start_time)), int(round(seg.end_time))])
        
        # Remove duplicates and sort
        gt_boundaries = sorted(set(gt_boundaries))
        pred_boundaries = sorted(set(pred_boundaries))
        
        # Count correct boundaries
        correct_boundaries = 0
        matched_pred = set()
        
        for gt_boundary in gt_boundaries:
            for i, pred_boundary in enumerate(pred_boundaries):
                if i not in matched_pred and abs(gt_boundary - pred_boundary) <= tolerance:
                    correct_boundaries += 1
                    matched_pred.add(i)
                    break
        
        precision = correct_boundaries / len(pred_boundaries) if pred_boundaries else 0.0
        recall = correct_boundaries / len(gt_boundaries) if gt_boundaries else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'boundary_precision': precision,
            'boundary_recall': recall,
            'boundary_f1': f1,
            'total_gt_boundaries': len(gt_boundaries),
            'total_pred_boundaries': len(pred_boundaries)
        }
    
    def multi_tolerance_boundary_analysis(self, gt_segments: List[Segment], pred_segments: List[Segment],
                                        tolerances: List[float] = [0.5, 1.0, 2.0, 3.0]) -> Dict[str, Dict[str, float]]:
        """Analyze boundary detection performance at multiple tolerance levels."""
        results = {}
        
        for tolerance in tolerances:
            boundary_metrics = self.boundary_detection_metrics(gt_segments, pred_segments, tolerance)
            results[f"tolerance_{tolerance}s"] = {
                'precision': boundary_metrics['boundary_precision'],
                'recall': boundary_metrics['boundary_recall'],
                'f1': boundary_metrics['boundary_f1']
            }
        
        return results
    
    def segmentation_quality_metrics(self, gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict[str, float]:
        """Calculate over-segmentation and under-segmentation errors."""
        over_seg_count = 0
        under_seg_count = 0
        
        # Over-segmentation: one GT segment matched by multiple predictions
        for gt_seg in gt_segments:
            matching_preds = 0
            for pred_seg in pred_segments:
                if self.calculate_iou(gt_seg, pred_seg) > 0.1:  # Small threshold for overlap
                    matching_preds += 1
            
            if matching_preds > 1:
                over_seg_count += 1
        
        # Under-segmentation: one prediction matched by multiple GT segments
        for pred_seg in pred_segments:
            matching_gts = 0
            for gt_seg in gt_segments:
                if self.calculate_iou(pred_seg, gt_seg) > 0.1:
                    matching_gts += 1
            
            if matching_gts > 1:
                under_seg_count += 1
        
        return {
            'over_segmentation_error': over_seg_count / len(gt_segments) if gt_segments else 0.0,
            'under_segmentation_error': under_seg_count / len(pred_segments) if pred_segments else 0.0,
            'over_segmented_segments': over_seg_count,
            'under_segmented_segments': under_seg_count
        }
    
    def temporal_consistency_analysis(self, gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict:
        """Analyze temporal consistency and duration patterns."""
        # Duration analysis per class
        gt_durations = defaultdict(list)
        pred_durations = defaultdict(list)
        
        for seg in gt_segments:
            duration = seg.end_time - seg.start_time
            gt_durations[seg.behavior].append(duration)
        
        for seg in pred_segments:
            duration = seg.end_time - seg.start_time
            pred_durations[seg.behavior].append(duration)
        
        duration_stats = {}
        all_behaviors = set(gt_durations.keys()) | set(pred_durations.keys())
        
        for behavior in all_behaviors:
            gt_durs = gt_durations[behavior]
            pred_durs = pred_durations[behavior]
            
            duration_stats[behavior] = {
                'gt_mean': np.mean(gt_durs) if gt_durs else 0,
                'gt_std': np.std(gt_durs) if gt_durs else 0,
                'gt_median': np.median(gt_durs) if gt_durs else 0,
                'pred_mean': np.mean(pred_durs) if pred_durs else 0,
                'pred_std': np.std(pred_durs) if pred_durs else 0,
                'pred_median': np.median(pred_durs) if pred_durs else 0,
                'gt_count': len(gt_durs),
                'pred_count': len(pred_durs),
                'duration_bias': (np.mean(pred_durs) - np.mean(gt_durs)) if (gt_durs and pred_durs) else 0
            }
        
        return {
            'duration_analysis': duration_stats
        }
    
    def generate_confusion_matrix(self, gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict[str, Dict[str, int]]:
        """Generate confusion matrix at second-level resolution."""
        # Find total duration
        max_time = max(max(int(seg.end_time) for seg in gt_segments), 
                      max(int(seg.end_time) for seg in pred_segments))
        
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        # Count second-by-second matches
        for second in range(max_time + 1):
            gt_behavior = self.get_behavior_at_second(gt_segments, second)
            pred_behavior = self.get_behavior_at_second(pred_segments, second)
            confusion_matrix[gt_behavior][pred_behavior] += 1
        
        return dict(confusion_matrix)
    
    def temporal_tolerance_analysis(self, gt_segments: List[Segment], pred_segments: List[Segment],
                                  tolerances: List[float] = [0.5, 1.0, 2.0, 3.0]) -> Dict[str, Dict[str, float]]:
        """Analyze performance at different temporal tolerances."""
        results = {}
        
        for tolerance in tolerances:
            # Find total duration
            max_time = max(max(int(seg.end_time) for seg in gt_segments), 
                          max(int(seg.end_time) for seg in pred_segments))
            
            correct = 0
            total = 0
            
            for second in range(max_time + 1):
                gt_behavior = self.get_behavior_at_second(gt_segments, second)
                
                # Check if prediction matches within tolerance window
                matches = False
                for window_offset in range(-int(tolerance), int(tolerance) + 1):
                    check_second = second + window_offset
                    if check_second >= 0:
                        pred_behavior = self.get_behavior_at_second(pred_segments, check_second)
                        if gt_behavior == pred_behavior:
                            matches = True
                            break
                
                if matches:
                    correct += 1
                total += 1
            
            accuracy = correct / total if total > 0 else 0.0
            results[f"tolerance_{tolerance}s"] = {
                'accuracy': accuracy,
                'correct_seconds': correct,
                'total_seconds': total
            }
        
        return results
    
    def evaluate_all(self, gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict:
        """Run comprehensive evaluation."""
        results = {}
        
        # Second-wise accuracy
        second_results = self.second_wise_accuracy(gt_segments, pred_segments)
        results.update(second_results)
        
        # IoU-based metrics
        iou_results = self.segment_iou_metrics(gt_segments, pred_segments)
        results.update(iou_results)
        
        # Edit distance
        results['edit_distance'] = self.edit_distance(gt_segments, pred_segments)
        
        # Boundary detection
        boundary_results = self.boundary_detection_metrics(gt_segments, pred_segments)
        results.update(boundary_results)
        
        # Multi-tolerance boundary analysis
        multi_boundary_results = self.multi_tolerance_boundary_analysis(gt_segments, pred_segments)
        results['boundary_tolerance_analysis'] = multi_boundary_results
        
        # Segmentation quality
        seg_quality = self.segmentation_quality_metrics(gt_segments, pred_segments)
        results.update(seg_quality)
        
        # Temporal consistency
        temporal_results = self.temporal_consistency_analysis(gt_segments, pred_segments)
        results.update(temporal_results)
        
        # Temporal tolerance analysis
        tolerance_results = self.temporal_tolerance_analysis(gt_segments, pred_segments)
        results['temporal_tolerance'] = tolerance_results
        
        # Confusion matrix
        results['confusion_matrix'] = self.generate_confusion_matrix(gt_segments, pred_segments)
        
        return results
    
    def print_results(self, results: Dict, output_path: Optional[Path] = None):
        """Print comprehensive evaluation results."""
        print("=" * 80)
        print("ACTION SEGMENTATION EVALUATION RESULTS (Second-Level)")
        print("=" * 80)
        
        # Core metrics
        print("\n📊 CORE METRICS")
        print("-" * 40)
        print(f"Second-wise Accuracy: {results['second_accuracy']:.4f}")
        print(f"Total Seconds Evaluated: {results['total_seconds']}")
        print(f"Edit Distance: {results['edit_distance']}")
        print(f"mAP (IoU 0.1-0.9): {results['mAP']:.4f}")
        
        # IoU-based metrics
        print("\n🎯 IoU-BASED SEGMENT METRICS")
        print("-" * 40)
        for threshold in [0.3, 0.5, 0.7]:
            print(f"IoU@{threshold} - Precision: {results[f'iou_{threshold}_precision']:.4f}, "
                  f"Recall: {results[f'iou_{threshold}_recall']:.4f}, "
                  f"F1: {results[f'iou_{threshold}_f1']:.4f}")
        
        # Boundary detection
        print("\n🎯 BOUNDARY DETECTION")
        print("-" * 40)
        print(f"Default Tolerance (±{self.tolerance}s):")
        print(f"  Precision: {results['boundary_precision']:.4f}")
        print(f"  Recall: {results['boundary_recall']:.4f}")
        print(f"  F1: {results['boundary_f1']:.4f}")
        print(f"  GT Boundaries: {results['total_gt_boundaries']}, Pred Boundaries: {results['total_pred_boundaries']}")
        
        print(f"\nBoundary Detection at Multiple Tolerances:")
        for tolerance_key, metrics in results['boundary_tolerance_analysis'].items():
            tolerance = tolerance_key.replace('tolerance_', '').replace('s', '')
            print(f"  ±{tolerance}s: Precision={metrics['precision']:.4f}, "
                  f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

        
        # Segmentation quality
        print("\n📏 SEGMENTATION QUALITY")
        print("-" * 40)
        print(f"Over-segmentation Error: {results['over_segmentation_error']:.4f} "
              f"({results['over_segmented_segments']} segments)")
        print(f"Under-segmentation Error: {results['under_segmentation_error']:.4f} "
              f"({results['under_segmented_segments']} segments)")
        
        # Temporal tolerance analysis
        print("\n⏱️  TEMPORAL TOLERANCE ANALYSIS")
        print("-" * 40)
        for tolerance_key, metrics in results['temporal_tolerance'].items():
            tolerance = tolerance_key.replace('tolerance_', '').replace('s', '')
            print(f"±{tolerance}s tolerance: {metrics['accuracy']:.4f} "
                  f"({metrics['correct_seconds']}/{metrics['total_seconds']})")
        
        # Per-class metrics
        print("\n📋 PER-CLASS METRICS")
        print("-" * 40)
        for class_name, metrics in results['per_class_metrics'].items():
            print(f"{class_name} (n={metrics['support']}):")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print(f"  F1: {metrics['f1']:.4f}")
        
        # Duration analysis
        print("\n⏱️  DURATION ANALYSIS")
        print("-" * 40)
        for behavior, stats in results['duration_analysis'].items():
            print(f"{behavior}:")
            print(f"  GT: {stats['gt_mean']:.1f}s ± {stats['gt_std']:.1f}s "
                  f"(median: {stats['gt_median']:.1f}s, n={stats['gt_count']})")
            print(f"  Pred: {stats['pred_mean']:.1f}s ± {stats['pred_std']:.1f}s "
                  f"(median: {stats['pred_median']:.1f}s, n={stats['pred_count']})")
            print(f"  Duration Bias: {stats['duration_bias']:+.1f}s")
        
        # Confusion matrix (simplified view for readability)
        print("\n🔄 CONFUSION MATRIX (seconds)")
        print("-" * 40)
        confusion_matrix = results['confusion_matrix']
        all_behaviors = sorted(set(confusion_matrix.keys()) | 
                             set(b for row in confusion_matrix.values() for b in row.keys()))
        
        # Only show non-background classes for clarity
        non_bg_behaviors = [b for b in all_behaviors if b != 'background']
        
        if non_bg_behaviors:
            # Print header
            print("GT\\Pred", end="\t")
            for pred in non_bg_behaviors:
                print(f"{pred[:8]:>8}", end="\t")
            print("background")
            
            # Print matrix
            for gt in non_bg_behaviors:
                print(f"{gt[:8]:8}", end="\t")
                for pred in non_bg_behaviors:
                    count = confusion_matrix.get(gt, {}).get(pred, 0)
                    print(f"{count:8}", end="\t")
                # Add background column
                bg_count = confusion_matrix.get(gt, {}).get('background', 0)
                print(f"{bg_count:8}")
        
        # Save results if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {output_path}")

def load_segments_from_json(filepath: str, start_segment: int = 0) -> List[Segment]:
    """Load segments from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON formats
    if isinstance(data, list):
        segments_data = data[start_segment:]
    elif 'segments' in data:
        segments_data = data['segments'][start_segment:]
    else:
        raise ValueError("Unknown JSON format")
    
    segments = []
    for seg_data in segments_data:
        # Handle different time formats
        if isinstance(seg_data, dict):
            start_time = seg_data.get('start_time', 0)
            end_time = seg_data.get('end_time', 0)
            behavior = seg_data.get('behavior', '')
        else:
            # Handle objects with attributes
            start_time = getattr(seg_data, 'start_time', 0)
            end_time = getattr(seg_data, 'end_time', 0)
            behavior = getattr(seg_data, 'behavior', '')
        
        # Convert time format if needed
        if isinstance(start_time, str):
            start_time = float(start_time.split(':')[0]) * 60 + float(start_time.split(':')[1])
        if isinstance(end_time, str):
            end_time = float(end_time.split(':')[0]) * 60 + float(end_time.split(':')[1])
        
        segments.append(Segment(start_time=start_time, end_time=end_time, behavior=behavior))
    
    return segments

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Action Segmentation Evaluation (Second-Level)')
    parser.add_argument('--ground-truth', required=True, 
                       help='Path to ground truth JSON file')
    parser.add_argument('--predictions', required=True, 
                       help='Path to predictions JSON file')
    parser.add_argument('--output-dir', default='./evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--start-segment', type=int, default=0,
                       help='Starting segment index (skip initial segments)')
    parser.add_argument('--tolerance', type=float, default=1.0,
                       help='Temporal tolerance for boundary detection (seconds)')
    
    args = parser.parse_args()
    
    # Load data
    gt_segments = load_segments_from_json(args.ground_truth, args.start_segment)
    pred_segments = load_segments_from_json(args.predictions)
    
    # Initialize evaluator
    evaluator = ActionSegmentationEvaluator(tolerance_seconds=args.tolerance)
    
    # Run evaluation
    results = evaluator.evaluate_all(gt_segments, pred_segments)
    
    # Print and save results
    output_path = Path(args.output_dir) / 'evaluation_results.json'
    evaluator.print_results(results, output_path)

if __name__ == "__main__":
    main()