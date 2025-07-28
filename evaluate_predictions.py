import json
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import mutual_info_score, matthews_corrcoef
import glob
import os
import re

from sklearn.preprocessing import LabelEncoder

# Add GCS imports for batch result downloading
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    print("Warning: google-cloud-storage not available. Batch result downloading will not work.")

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
        """Calculate second-wise accuracy with detailed per-class metrics, plus mutual info and Pearson correlation between true labels and correctness."""
        # Find total duration
        max_time = max(max(int(seg.end_time) for seg in gt_segments), 
                      max(int(seg.end_time) for seg in pred_segments))
        
        # Create second-by-second labels
        gt_labels = []
        pred_labels = []
        corrects = []
        for second in range(max_time + 1):
            gt_label = self.get_behavior_at_second(gt_segments, second)
            pred_label = self.get_behavior_at_second(pred_segments, second)
            gt_labels.append(gt_label)
            pred_labels.append(pred_label)
            corrects.append(int(gt_label == pred_label))
        
        # Calculate overall accuracy
        correct = sum(corrects)
        total = len(gt_labels)

        # For mutual info: encode string labels to integers
        le = LabelEncoder()
        all_labels = gt_labels + pred_labels
        le.fit(all_labels)
        gt_labels_int = le.transform(gt_labels)
        pred_labels_int = le.transform(pred_labels)
        
        mutual_info = mutual_info_score(gt_labels_int, pred_labels_int)
        
        
        MCC = matthews_corrcoef(gt_labels_int, pred_labels_int)
        
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
        
        # Calculate per-class precision, recall, f1, balanced accuracy
        per_class_results = {}
        f1_scores = []
        supports = []
        
        for cls in all_classes:
            tp = class_metrics[cls]['tp']
            fp = class_metrics[cls]['fp']
            fn = class_metrics[cls]['fn']
            tn = class_metrics[cls]['tn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Balanced accuracy
            sensitivity = recall  # Same as recall
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            balanced_acc = (sensitivity + specificity) / 2
            
            support = tp + fn  # Number of true instances
            
            per_class_results[cls] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'balanced_accuracy': balanced_acc,
                'support': support
            }
            
            f1_scores.append(f1)
            supports.append(support)
        
        # Calculate macro and weighted F1
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0
        
        total_support = sum(supports)
        weighted_f1 = 0.0
        if total_support > 0:
            for cls in all_classes:
                weight = per_class_results[cls]['support'] / total_support
                weighted_f1 += weight * per_class_results[cls]['f1']
        
        return {
            'second_accuracy': correct / total,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_metrics': per_class_results,
            'total_seconds': total,
            'mutual_info_gt_vs_pred': mutual_info,
            'MCC': MCC
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
        recalls = [0]
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
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            ap_scores.append(precision * (recall - recalls[-1]))
            recalls.append(recall)
        
        results['mAP'] = np.sum(ap_scores)
        
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
    
    def calculate_temporal_coverage(self, gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict[str, Dict[str, float]]:
        """Calculate temporal coverage analysis for each behavior class."""
        coverage = {}
        
        # Find total video duration
        total_duration = max(max(seg.end_time for seg in gt_segments), 
                           max(seg.end_time for seg in pred_segments))
        
        # Get all unique behaviors
        all_behaviors = set(seg.behavior for seg in gt_segments) | set(seg.behavior for seg in pred_segments)
        
        for behavior in all_behaviors:
            # Calculate ground truth duration for this behavior
            gt_duration = sum(seg.end_time - seg.start_time 
                             for seg in gt_segments if seg.behavior == behavior)
            
            # Calculate predicted duration for this behavior
            pred_duration = sum(seg.end_time - seg.start_time 
                               for seg in pred_segments if seg.behavior == behavior)
            
            # Calculate percentages and ratios
            gt_percentage = (gt_duration / total_duration * 100) if total_duration > 0 else 0
            pred_percentage = (pred_duration / total_duration * 100) if total_duration > 0 else 0
            coverage_ratio = (pred_duration / gt_duration) if gt_duration > 0 else 0
            duration_error = pred_duration - gt_duration
            
            coverage[behavior] = {
                'gt_duration': gt_duration,
                'pred_duration': pred_duration,
                'gt_percentage': gt_percentage,
                'pred_percentage': pred_percentage,
                'coverage_ratio': coverage_ratio,
                'duration_error': duration_error,
                'duration_error_percentage': (duration_error / gt_duration * 100) if gt_duration > 0 else 0
            }
        
        return coverage
    
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
                'gt_std_error': np.std(gt_durs) / np.sqrt(len(gt_durs)-1) if gt_durs else 0,
                'gt_median': np.median(gt_durs) if gt_durs else 0,
                'pred_mean': np.mean(pred_durs) if pred_durs else 0,
                'pred_std_error': np.std(pred_durs) / np.sqrt(len(pred_durs)-1) if pred_durs else 0,
                'pred_median': np.median(pred_durs) if pred_durs else 0,
                'gt_count': len(gt_durs),
                'pred_count': len(pred_durs),
                'duration_bias': (np.mean(pred_durs) - np.mean(gt_durs)) if (gt_durs and pred_durs) else 0
            }
        
        return {
            'duration_analysis': duration_stats
        }
    
    def calculate_behavior_fraction_correlation(self, gt_segments: List[Segment], pred_segments: List[Segment], 
                                             chunk_size_seconds: float = 10.0) -> Dict:
        """Calculate correlation between ground truth and predicted behavior fractions in time chunks."""
        # Find total duration
        total_duration = max(max(seg.end_time for seg in gt_segments), 
                           max(seg.end_time for seg in pred_segments))
        
        # Get all unique behaviors
        all_behaviors = set(seg.behavior for seg in gt_segments) | set(seg.behavior for seg in pred_segments)
        
        # Initialize lists to store fractions for each chunk
        gt_fractions = defaultdict(list)
        pred_fractions = defaultdict(list)
        
        
        # Process each chunk
        for chunk_start in np.arange(0, total_duration, chunk_size_seconds):
            chunk_end = chunk_start + chunk_size_seconds
            
            # Calculate fractions for ground truth
            gt_chunk_durations = defaultdict(float)
            for seg in gt_segments:
                if seg.end_time <= chunk_start or seg.start_time >= chunk_end:
                    continue
                overlap_start = max(seg.start_time, chunk_start)
                overlap_end = min(seg.end_time, chunk_end)
                gt_chunk_durations[seg.behavior] += overlap_end - overlap_start
            
            # Calculate fractions for predictions
            pred_chunk_durations = defaultdict(float)
            for seg in pred_segments:
                if seg.end_time <= chunk_start or seg.start_time >= chunk_end:
                    continue
                overlap_start = max(seg.start_time, chunk_start)
                overlap_end = min(seg.end_time, chunk_end)
                pred_chunk_durations[seg.behavior] += overlap_end - overlap_start
            
            # Convert durations to fractions
            gt_total = sum(gt_chunk_durations.values())
            pred_total = sum(pred_chunk_durations.values())
            
            for behavior in all_behaviors:
                gt_fractions[behavior].append(gt_chunk_durations[behavior] / gt_total if gt_total > 0 else 0)
                pred_fractions[behavior].append(pred_chunk_durations[behavior] / pred_total if pred_total > 0 else 0)
        
        # Calculate correlations for each behavior
        correlations = {}
        for behavior in all_behaviors:
            if len(gt_fractions[behavior]) > 1:  # Need at least 2 points for correlation
                correlation = np.corrcoef(gt_fractions[behavior], pred_fractions[behavior])[0, 1]
                correlations[behavior] = correlation
            else:
                correlations[behavior] = float('nan')
        
        # Calculate mean correlation across behaviors
        valid_correlations = [c for c in correlations.values() if not np.isnan(c)]
        mean_correlation = np.mean(valid_correlations) if valid_correlations else float('nan')
        
        return {
            'per_behavior_correlation': correlations,
            'mean_correlation': mean_correlation,
            'chunk_size_seconds': chunk_size_seconds,
            'num_chunks': len(gt_fractions[list(all_behaviors)[0]]) if all_behaviors else 0
        }
    
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
        
        # Temporal coverage analysis
        coverage_results = self.calculate_temporal_coverage(gt_segments, pred_segments)
        results['temporal_coverage'] = coverage_results
        
        # Temporal tolerance analysis
        tolerance_results = self.temporal_tolerance_analysis(gt_segments, pred_segments)
        results['temporal_tolerance'] = tolerance_results
        
        # Behavior fraction correlation
        behavior_fraction_results = self.calculate_behavior_fraction_correlation(gt_segments, pred_segments)
        results['behavior_fraction_correlation'] = behavior_fraction_results
        
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
        print(f"Macro F1 (unweighted): {results['macro_f1']:.4f}")
        print(f"Weighted F1 (by frequency): {results['weighted_f1']:.4f}")
        print(f"Total Seconds Evaluated: {results['total_seconds']}")
        print(f"Edit Distance: {results['edit_distance']}")
        print(f"mAP (IoU 0.1-0.9): {results['mAP']:.4f}")
        
        # Total time summary
        print("\n⏱️  TOTAL TIME SUMMARY")
        print("-" * 40)
        print(f"Total Video Duration: {results['total_seconds']} seconds")
        
        # Calculate total time for each behavior
        if 'temporal_coverage' in results:
            print("\n📈 BEHAVIOR DURATION BREAKDOWN:")
            print("-" * 40)
            
            # Sort behaviors by ground truth duration (descending)
            sorted_behaviors = sorted(
                results['temporal_coverage'].items(),
                key=lambda x: x[1]['gt_duration'],
                reverse=True
            )
            
            for behavior, coverage in sorted_behaviors:
                gt_duration = coverage['gt_duration']
                pred_duration = coverage['pred_duration']
                gt_percentage = coverage['gt_percentage']
                pred_percentage = coverage['pred_percentage']
                
                print(f"{behavior}:")
                print(f"  Ground Truth: {gt_duration:.1f}s ({gt_percentage:.1f}% of video)")
                print(f"  Predicted:    {pred_duration:.1f}s ({pred_percentage:.1f}% of video)")
                print(f"  Difference:   {pred_duration - gt_duration:+.1f}s")
        
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
            print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        
        # Temporal coverage analysis
        print("\n📊 TEMPORAL COVERAGE ANALYSIS")
        print("-" * 40)
        for behavior, coverage in results['temporal_coverage'].items():
            print(f"{behavior}:")
            print(f"  GT: {coverage['gt_duration']:.1f}s ({coverage['gt_percentage']:.1f}% of video)")
            print(f"  Pred: {coverage['pred_duration']:.1f}s ({coverage['pred_percentage']:.1f}% of video)")
            print(f"  Coverage Ratio: {coverage['coverage_ratio']:.2f}")
            print(f"  Duration Error: {coverage['duration_error']:+.1f}s ({coverage['duration_error_percentage']:+.1f}%)")
        
        # Duration analysis
        print("\n⏱️  DURATION ANALYSIS")
        print("-" * 40)
        for behavior, stats in results['duration_analysis'].items():
            print(f"{behavior}:")
            print(f"  GT: {stats['gt_mean']:.1f}s ± {stats['gt_std_error']:.1f}s "
                  f"(median: {stats['gt_median']:.1f}s, n={stats['gt_count']})")
            print(f"  Pred: {stats['pred_mean']:.1f}s ± {stats['pred_std_error']:.1f}s "
                  f"(median: {stats['pred_median']:.1f}s, n={stats['pred_count']})")
            print(f"  Duration Bias: {stats['duration_bias']:+.1f}s")
        
        
            
        
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

def find_matching_files(gt_dir: str, pred_dir: str) -> List[Tuple[str, str]]:
    """Find matching ground truth and prediction files in directories."""
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)
    
    # Get all JSON files in both directories
    gt_files = list(gt_path.glob("*.json"))
    pred_files = list(pred_path.glob("*.json"))
    
    # Create a mapping of filenames to full paths
    gt_file_map = {f.stem: f for f in gt_files}
    pred_file_map = {f.stem: f for f in pred_files}
    
    # Find matching pairs
    matching_pairs = []
    for filename in gt_file_map.keys():
        if filename in pred_file_map:
            matching_pairs.append((str(gt_file_map[filename]), str(pred_file_map[filename])))
    
    # Save ground truth file paths for visualization in the output directory
    gt_file_paths = [gt for gt, _ in matching_pairs]
    output_path = Path(gt_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    gt_paths_file = output_path / 'gt_file_paths.json'
    with open(gt_paths_file, 'w') as f:
        json.dump(gt_file_paths, f, indent=2)
    print(f"\n💾 Saved ground truth file paths to: {gt_paths_file}")
    
    return matching_pairs

def evaluate_single_file(gt_file: str, pred_file: str, evaluator: ActionSegmentationEvaluator, 
                        start_segment: int = 0) -> Tuple[str, Dict]:
    """Evaluate a single pair of ground truth and prediction files."""
    print(f"\n📁 Evaluating: {Path(gt_file).name} vs {Path(pred_file).name}")
    
    # Load data
    gt_segments = load_segments_from_json(gt_file, start_segment)
    pred_segments = load_segments_from_json(pred_file)
    
    print(f"✅ Loaded {len(gt_segments)} ground truth segments, {len(pred_segments)} predicted segments")
    
    # Run evaluation
    results = evaluator.evaluate_all(gt_segments, pred_segments)
    
    return Path(gt_file).stem, results

def aggregate_results(all_results: List[Tuple[str, Dict]]) -> Dict:
    """Aggregate results from multiple files."""
    if not all_results:
        return {}
    
    # Initialize aggregated metrics
    aggregated = {
        'file_count': len(all_results),
        'files': {},
        'summary': {
            'second_accuracy': [],
            'macro_f1': [],
            'weighted_f1': [],
            'mAP': [],
            'mutual_info_gt_vs_pred': [],
            'MCC': [],
            'edit_distance': [],
            'boundary_f1': [],
            'over_segmentation_error': [],
            'under_segmentation_error': []
        }
    }
    
    # Collect results from each file
    for filename, results in all_results:
        aggregated['files'][filename] = results
        
        # Add to summary lists
        for metric in aggregated['summary'].keys():
            if metric in results:
                aggregated['summary'][metric].append(results[metric])
    
    # Calculate summary statistics
    metrics_to_process = list(aggregated['summary'].keys())
    for metric in metrics_to_process:
        values = aggregated['summary'][metric]
        if values:
            aggregated['summary'][f'{metric}_mean'] = np.mean(values)
            aggregated['summary'][f'{metric}_std_error'] = np.std(values) / np.sqrt(len(values)-1)
            aggregated['summary'][f'{metric}_min'] = np.min(values)
            aggregated['summary'][f'{metric}_max'] = np.max(values)
    
    return aggregated

def print_aggregated_results(aggregated_results: Dict):
    """Print aggregated results from multiple files."""
    print("=" * 80)
    print("AGGREGATED EVALUATION RESULTS")
    print("=" * 80)
    
    print(f"\n📊 SUMMARY STATISTICS ({aggregated_results['file_count']} files)")
    print("-" * 50)
    
    summary = aggregated_results['summary']
    print(f"Second-wise Accuracy: {summary['second_accuracy_mean']:.4f} ± {summary['second_accuracy_std_error']:.4f} "
          f"[{summary['second_accuracy_min']:.4f}, {summary['second_accuracy_max']:.4f}]")
    print(f"Macro F1: {summary['macro_f1_mean']:.4f} ± {summary['macro_f1_std_error']:.4f} "
          f"[{summary['macro_f1_min']:.4f}, {summary['macro_f1_max']:.4f}]")
    print(f"Weighted F1: {summary['weighted_f1_mean']:.4f} ± {summary['weighted_f1_std_error']:.4f} "
          f"[{summary['weighted_f1_min']:.4f}, {summary['weighted_f1_max']:.4f}]")
    print(f"mAP: {summary['mAP_mean']:.4f} ± {summary['mAP_std_error']:.4f} "
          f"[{summary['mAP_min']:.4f}, {summary['mAP_max']:.4f}]")
    print(f"Mutual Info: {summary['mutual_info_gt_vs_pred_mean']:.4f} ± {summary['mutual_info_gt_vs_pred_std_error']:.4f} "
          f"[{summary['mutual_info_gt_vs_pred_min']:.4f}, {summary['mutual_info_gt_vs_pred_max']:.4f}]")
    print(f"MCC: {summary['MCC_mean']:.4f} ± {summary['MCC_std_error']:.4f} "
          f"[{summary['MCC_min']:.4f}, {summary['MCC_max']:.4f}]")
    print(f"Boundary F1: {summary['boundary_f1_mean']:.4f} ± {summary['boundary_f1_std_error']:.4f} "
          f"[{summary['boundary_f1_min']:.4f}, {summary['boundary_f1_max']:.4f}]")
    print(f"Edit Distance: {summary['edit_distance_mean']:.1f} ± {summary['edit_distance_std_error']:.1f} "
          f"[{summary['edit_distance_min']:.0f}, {summary['edit_distance_max']:.0f}]")
    
    # Total time summary across all files
    print(f"\n⏱️  TOTAL TIME SUMMARY ACROSS ALL FILES")
    print("-" * 50)
    
    # Calculate total durations across all files
    total_gt_durations = {}
    total_pred_durations = {}
    total_video_duration = 0
    
    for filename, results in aggregated_results['files'].items():
        total_video_duration += results['total_seconds']
        
        if 'temporal_coverage' in results:
            for behavior, coverage in results['temporal_coverage'].items():
                if behavior not in total_gt_durations:
                    total_gt_durations[behavior] = 0
                    total_pred_durations[behavior] = 0
                
                total_gt_durations[behavior] += coverage['gt_duration']
                total_pred_durations[behavior] += coverage['pred_duration']
    
    print(f"Total Video Duration Across All Files: {total_video_duration:.1f} seconds")
    
    if total_gt_durations:
        print(f"\n📈 BEHAVIOR DURATION BREAKDOWN (ACROSS ALL FILES):")
        print("-" * 50)
        
        # Sort behaviors by total ground truth duration (descending)
        sorted_behaviors = sorted(
            total_gt_durations.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for behavior, gt_duration in sorted_behaviors:
            pred_duration = total_pred_durations[behavior]
            gt_percentage = (gt_duration / total_video_duration * 100) if total_video_duration > 0 else 0
            pred_percentage = (pred_duration / total_video_duration * 100) if total_video_duration > 0 else 0
            
            print(f"{behavior}:")
            print(f"  Ground Truth: {gt_duration:.1f}s ({gt_percentage:.1f}% of total)")
            print(f"  Predicted:    {pred_duration:.1f}s ({pred_percentage:.1f}% of total)")
            print(f"  Difference:   {pred_duration - gt_duration:+.1f}s")
    
    print(f"\n📋 PER-FILE RESULTS")
    print("-" * 50)
    for filename, results in aggregated_results['files'].items():
        print(f"{filename}: Acc={results['second_accuracy']:.3f}, "
              f"F1={results['weighted_f1']:.3f}, mAP={results['mAP']:.3f}, "
              f"MI={results['mutual_info_gt_vs_pred']:.3f}, MCC={results['MCC']:.3f}")
        
        # Add per-file time information
        if 'temporal_coverage' in results:
            print(f"  Duration: {results['total_seconds']}s")
            for behavior, coverage in results['temporal_coverage'].items():
                print(f"    {behavior}: GT={coverage['gt_duration']:.1f}s, Pred={coverage['pred_duration']:.1f}s")

def download_batch_results(gcs_output_dir: str, local_output_dir: str) -> bool:
    """
    Download all files from a GCS directory to a local directory.
    
    Args:
        gcs_output_dir: GCS URI of the directory containing batch results
        local_output_dir: Local directory to save the files
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not GCS_AVAILABLE:
        print("Error: google-cloud-storage library not available")
        return False
        
    print(f"Looking for files in GCS directory: {gcs_output_dir}")
    if not gcs_output_dir.endswith('/'):
        gcs_output_dir += '/'
    if not os.path.exists(local_output_dir):
        os.makedirs(local_output_dir, exist_ok=True)
        
    try:
        client = storage.Client()
        bucket_name, prefix = gcs_output_dir[5:].split('/', 1)
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        
        file_count = 0
        for blob in blobs:
            # Only download files, not directories
            if not blob.name.endswith('/'):
                file_count += 1
                local_path = os.path.join(local_output_dir, os.path.basename(blob.name))
                print(f"Downloading {blob.name} to {local_path}")
                blob.download_to_filename(local_path)
                
        if file_count == 0:
            print(f"No files found in {gcs_output_dir}")
            return False
        else:
            print(f"Downloaded {file_count} files to {local_output_dir}")
            return True
            
    except Exception as e:
        print(f"Error downloading batch results: {str(e)}")
        return False

def split_jsonl_predictions(jsonl_path: str, output_dir: str):
    """Split a .jsonl predictions file into per-video JSONs (old format)."""
    os.makedirs(output_dir, exist_ok=True)
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            # Extract video filename
            try:
                file_uri = obj['request']['contents'][0]['parts'][1]['fileData']['fileUri']
                # e.g. gs://videos_freezing_mice/calms/mouse003.mp4
                video_name = os.path.splitext(os.path.basename(file_uri))[0]  # mouse003
            except Exception as e:
                print(f"Skipping line due to missing fileUri: {e}")
                continue
            # Extract segments JSON string
            try:
                text = obj['response']['candidates'][0]['content']['parts'][0]['text']
                # Remove markdown code block if present
                match = re.search(r'```json\n(.*)\n```', text, re.DOTALL)
                if match:
                    segments_json = match.group(1)
                else:
                    segments_json = text
                segments_data = json.loads(segments_json)
                segments = segments_data['segments'] if 'segments' in segments_data else segments_data
            except Exception as e:
                print(f"Skipping {video_name} due to segment parse error: {e}")
                continue
            # Write to file (list of segments, not wrapped in dict)
            out_path = os.path.join(output_dir, f"{video_name}.json")
            with open(out_path, 'w') as out_f:
                json.dump(segments, out_f, indent=2)
            print(f"Wrote {out_path}")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Action Segmentation Evaluation (Second-Level)')
    parser.add_argument('--ground-truth', required=True, 
                       help='Path to ground truth directory or single JSON file')
    parser.add_argument('--predictions', required=True, 
                       help='Path to predictions directory or single JSON file')
    parser.add_argument('--output-dir', default='./evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--start-segment', type=int, default=0,
                       help='Starting segment index (skip initial segments)')
    parser.add_argument('--tolerance', type=float, default=1.0,
                       help='Temporal tolerance for boundary detection (seconds)')
    parser.add_argument('--single-file', action='store_true',
                       help='Treat inputs as single files instead of directories')
    parser.add_argument('--download-batch-results', help='GCS URI to download batch results from (e.g., gs://bucket/batch_results/calms/)')
    parser.add_argument('--batch-results-dir', help='Local directory to save downloaded batch results (defaults to --predictions if not specified)')
    parser.add_argument('--split-jsonl', action='store_true', help='Split a .jsonl predictions file into per-video JSONs and exit')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ActionSegmentationEvaluator(tolerance_seconds=args.tolerance)
    
    # Download batch results if requested
    if args.download_batch_results:
        if not args.batch_results_dir:
            args.batch_results_dir = args.predictions
        print(f"Downloading batch results from {args.download_batch_results} to {args.batch_results_dir}")
        if download_batch_results(args.download_batch_results, args.batch_results_dir):
            print("Batch results downloaded successfully")
            # Update predictions path to use downloaded results
            args.predictions = args.batch_results_dir
        else:
            print("Failed to download batch results")
            return 1
    
    if args.split_jsonl:
        # If predictions is a .jsonl file, split it
        if args.predictions.endswith('.jsonl'):
            out_dir = os.path.join(os.path.dirname(args.predictions), 'split_predictions')
            split_jsonl_predictions(args.predictions, out_dir)
            print(f"\nAll files written to {out_dir}")
        else:
            print("--split-jsonl requires --predictions to be a .jsonl file")
        return

    # --- NEW: Auto-handle .jsonl predictions ---
    predictions_path = args.predictions
    
    # Check if predictions_path is a directory that contains a .jsonl file
    if os.path.isdir(predictions_path):
        jsonl_files = [f for f in os.listdir(predictions_path) if f.endswith('.jsonl')]
        if jsonl_files:
            # Prioritize predictions.jsonl over chunked files
            if 'predictions.jsonl' in jsonl_files:
                jsonl_path = os.path.join(predictions_path, 'predictions.jsonl')
            else:
                # Use the first .jsonl file found
                jsonl_path = os.path.join(predictions_path, jsonl_files[0])
            split_dir = os.path.join(predictions_path, 'split_predictions')
            # Only split if directory does not exist or is empty
            if not os.path.exists(split_dir) or not os.listdir(split_dir):
                print(f"Splitting {jsonl_path} to {split_dir}...")
                split_jsonl_predictions(jsonl_path, split_dir)
            else:
                print(f"Using existing split predictions in {split_dir}")
            predictions_path = split_dir
    elif predictions_path.endswith('.jsonl'):
        split_dir = os.path.join(os.path.dirname(predictions_path), 'split_predictions')
        # Only split if directory does not exist or is empty
        if not os.path.exists(split_dir) or not os.listdir(split_dir):
            print(f"Splitting {predictions_path} to {split_dir}...")
            split_jsonl_predictions(predictions_path, split_dir)
        else:
            print(f"Using existing split predictions in {split_dir}")
        predictions_path = split_dir
    # --- END NEW ---

    if args.single_file:
        # Single file evaluation (original behavior)
        print(f"\U0001F4C1 Loading ground truth from: {args.ground_truth}")
        gt_segments = load_segments_from_json(args.ground_truth, args.start_segment)
        print(f"✅ Loaded {len(gt_segments)} ground truth segments")
        print(f"\U0001F4C1 Loading predictions from: {predictions_path}")
        pred_segments = load_segments_from_json(predictions_path)
        print(f"✅ Loaded {len(pred_segments)} predicted segments")
        # Run evaluation
        print(f"\n🔄 Running comprehensive evaluation...")
        results = evaluator.evaluate_all(gt_segments, pred_segments)
        # Print and save results
        output_path = Path(args.output_dir) / 'evaluation_results.json'
        evaluator.print_results(results, output_path)
        print(f"\n🎉 Evaluation complete!")
        print(f"📊 Key metrics: Accuracy={results['second_accuracy']:.3f}, "
              f"Weighted F1={results['weighted_f1']:.3f}, mAP={results['mAP']:.3f}, "
              f"Mutual Info={results['mutual_info_gt_vs_pred']:.3f}, MCC={results['MCC']:.3f}")
    else:
        # Directory evaluation (new behavior)
        print(f"\U0001F4C1 Processing directories:")
        print(f"  Ground truth: {args.ground_truth}")
        print(f"  Predictions: {predictions_path}")
        # Find matching files
        matching_pairs = find_matching_files(args.ground_truth, predictions_path)
        if not matching_pairs:
            print("❌ No matching files found in the directories!")
            print("Make sure both directories contain JSON files with matching names (excluding extension)")
            return
        print(f"✅ Found {len(matching_pairs)} matching file pairs")
        # Evaluate each pair
        all_results = []
        for gt_file, pred_file in matching_pairs:
            try:
                filename, results = evaluate_single_file(gt_file, pred_file, evaluator, args.start_segment)
                all_results.append((filename, results))
            except Exception as e:
                print(f"❌ Error evaluating {Path(gt_file).name}: {e}")
                continue
        if not all_results:
            print("❌ No files were successfully evaluated!")
            return
        # Aggregate results
        print(f"\n🔄 Aggregating results from {len(all_results)} files...")
        aggregated_results = aggregate_results(all_results)
        # Print aggregated results
        print_aggregated_results(aggregated_results)
        # Save results
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        # Save ground truth file paths for visualization in the output directory
        gt_file_paths = [gt for gt, _ in matching_pairs]
        gt_paths_file = output_path / 'gt_file_paths.json'
        with open(gt_paths_file, 'w') as f:
            json.dump(gt_file_paths, f, indent=2)
        print(f"\n💾 Saved ground truth file paths to: {gt_paths_file}")
        # Save aggregated results
        aggregated_output = output_path / 'aggregated_results.json'
        with open(aggregated_output, 'w') as f:
            json.dump(aggregated_results, f, indent=2, default=str)
        # Save individual results
        individual_output = output_path / 'individual_results'
        individual_output.mkdir(exist_ok=True)
        for filename, results in all_results:
            file_output = individual_output / f'{filename}_results.json'
            with open(file_output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to:")
        print(f"  Aggregated: {aggregated_output}")
        print(f"  Individual: {individual_output}/")
        print(f"\n🎉 Batch evaluation complete!")
        print(f"📊 Average metrics: Accuracy={aggregated_results['summary']['second_accuracy_mean']:.3f}, "
              f"F1={aggregated_results['summary']['weighted_f1_mean']:.3f}, "
              f"mAP={aggregated_results['summary']['mAP_mean']:.3f}")

if __name__ == "__main__":
    main()