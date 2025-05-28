import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Segment:
    start_time: float
    end_time: float
    behavior: str

def calculate_iou(seg1: Segment, seg2: Segment) -> float:
    """Calculate Intersection over Union between two segments."""
    intersection_start = max(seg1.start_time, seg2.start_time)
    intersection_end = min(seg1.end_time, seg2.end_time)
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection_duration = intersection_end - intersection_start
    union_duration = (seg1.end_time - seg1.start_time) + (seg2.end_time - seg2.start_time) - intersection_duration
    
    return intersection_duration / union_duration if union_duration > 0 else 0.0

def calculate_edit_distance(gt_segments: List[Segment], pred_segments: List[Segment]) -> int:
    """Calculate edit distance between ground truth and predicted segments."""
    # Convert segments to strings of behaviors
    gt_string = "".join(seg.behavior for seg in gt_segments)
    pred_string = "".join(seg.behavior for seg in pred_segments)
    
    # Initialize matrix
    m, n = len(gt_string), len(pred_string)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill matrix
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
        
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_string[i-1] == pred_string[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j-1] + 1,  # substitution
                             dp[i-1][j] + 1,      # deletion
                             dp[i][j-1] + 1)      # insertion
    
    return dp[m][n]

def calculate_boundary_metrics(gt_segments: List[Segment], pred_segments: List[Segment], 
                             tolerance: float = 1.0) -> Dict[str, float]:
    """Calculate boundary precision and recall."""
    gt_boundaries = [(seg.start_time, seg.end_time) for seg in gt_segments]
    pred_boundaries = [(seg.start_time, seg.end_time) for seg in pred_segments]
    
    # Count correct boundaries
    correct_boundaries = 0
    for gt_start, gt_end in gt_boundaries:
        for pred_start, pred_end in pred_boundaries:
            if (abs(gt_start - pred_start) <= tolerance and 
                abs(gt_end - pred_end) <= tolerance):
                correct_boundaries += 1
                break
    
    # Calculate precision and recall
    precision = correct_boundaries / len(pred_boundaries) if pred_boundaries else 0.0
    recall = correct_boundaries / len(gt_boundaries) if gt_boundaries else 0.0
    
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    }

def calculate_segment_f1(gt_segments: List[Segment], pred_segments: List[Segment], 
                        iou_threshold: float = 0.5) -> Dict[str, float]:
    """Calculate segment-level F1 score."""
    true_positives = 0
    false_positives = len(pred_segments)
    false_negatives = len(gt_segments)
    
    for gt_seg in gt_segments:
        for pred_seg in pred_segments:
            if gt_seg.behavior == pred_seg.behavior:
                iou = calculate_iou(gt_seg, pred_seg)
                if iou >= iou_threshold:
                    true_positives += 1
                    false_positives -= 1
                    false_negatives -= 1
                    break
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        f"segment_f1@{iou_threshold}": f1,
        "segment_precision": precision,
        "segment_recall": recall
    }

def calculate_segmentation_errors(gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict[str, float]:
    """Calculate over-segmentation and under-segmentation errors."""
    over_segmentation = 0
    under_segmentation = 0
    
    # For each ground truth segment, count how many predicted segments overlap with it
    for gt_seg in gt_segments:
        overlapping_preds = 0
        for pred_seg in pred_segments:
            if calculate_iou(gt_seg, pred_seg) > 0:
                overlapping_preds += 1
        
        if overlapping_preds > 1:
            over_segmentation += 1
    
    # For each predicted segment, count how many ground truth segments overlap with it
    for pred_seg in pred_segments:
        overlapping_gt = 0
        for gt_seg in gt_segments:
            if calculate_iou(pred_seg, gt_seg) > 0:
                overlapping_gt += 1
        
        if overlapping_gt > 1:
            under_segmentation += 1
    
    return {
        "over_segmentation_error": over_segmentation / len(gt_segments) if gt_segments else 0.0,
        "under_segmentation_error": under_segmentation / len(pred_segments) if pred_segments else 0.0
    }

def calculate_all_metrics(gt_segments: List[Segment], pred_segments: List[Segment]) -> Dict[str, float]:
    """Calculate all advanced metrics for action segmentation."""
    metrics = {}
    
    # Calculate IoU at different thresholds
    iou_thresholds = [0.3, 0.5, 0.7]
    for threshold in iou_thresholds:
        metrics.update(calculate_segment_f1(gt_segments, pred_segments, threshold))
    
    # Calculate boundary metrics
    metrics.update(calculate_boundary_metrics(gt_segments, pred_segments))
    
    # Calculate edit distance
    metrics["edit_distance"] = calculate_edit_distance(gt_segments, pred_segments)
    
    # Calculate segmentation errors
    metrics.update(calculate_segmentation_errors(gt_segments, pred_segments))
    
    return metrics 