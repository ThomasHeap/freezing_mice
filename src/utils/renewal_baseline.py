from typing import List, Dict, Tuple
import numpy as np
from collections import Counter
from evaluate_predictions import Segment, ActionSegmentationEvaluator

def simulate_renewal_random_segments(gt_segments: List[Segment], video_length: float, n_runs: int = 5) -> List[List[Segment]]:
    """
    Simulate random segmentations using a marked renewal process.
    Args:
        gt_segments: List of ground truth Segment objects
        video_length: Total duration of the video (in seconds)
        n_runs: Number of random segmentations to generate
    Returns:
        List of simulated segment lists (one per run)
    """
    # Extract empirical segment length distributions and class frequencies
    class_lengths = {}
    class_counts = Counter()
    for seg in gt_segments:
        c = seg.behavior
        l = seg.end_time - seg.start_time
        if c not in class_lengths:
            class_lengths[c] = []
        class_lengths[c].append(l)
        class_counts[c] += 1
    n_classes = len(class_counts)
    total_segments = sum(class_counts.values())
    class_probs = {c: class_counts[c] / total_segments for c in class_counts}
    class_list = list(class_counts.keys())

    all_simulated = []
    for _ in range(n_runs):
        t = 0.0
        segments = []
        seg_num = 0
        while t < video_length:
            c = np.random.choice(class_list, p=[class_probs[k] for k in class_list])
            l = float(np.random.choice(class_lengths[c]))
            l = min(l, video_length - t)
            segments.append(Segment(start_time=t, end_time=t+l, behavior=c))
            t += l
            seg_num += 1
        all_simulated.append(segments)
    return all_simulated

def compute_renewal_baseline_metrics(gt_segments: List[Segment], video_length: float, n_runs: int = 5) -> Dict[str, Tuple[float, float]]:
    """
    Compute the renewal process baseline for all metrics by simulating random segmentations and evaluating them.
    Returns a dict of metric_name: (mean, std)
    """
    evaluator = ActionSegmentationEvaluator()
    all_metrics = []
    print("Computing renewal baseline metrics...")
    simulated_sets = simulate_renewal_random_segments(gt_segments, video_length, n_runs)
    print(f"Simulated {len(simulated_sets)} random segmentations")
    metrics = {}
    for pred_segments in simulated_sets:
        metrics['mAP'] = evaluator.segment_iou_metrics(gt_segments, pred_segments)['mAP']
        core_metrics = evaluator.second_wise_accuracy(gt_segments, pred_segments)
        metrics['second_accuracy'] = core_metrics['second_accuracy']
        metrics['macro_f1'] = core_metrics['macro_f1']
        metrics['mutual_info_gt_vs_pred'] = core_metrics['mutual_info_gt_vs_pred']
        metrics['MCC'] = core_metrics['MCC']
        all_metrics.append(metrics)
        
    # Aggregate results
    metric_names = list(all_metrics[0].keys()) if all_metrics else []
    results = {}
    for k in metric_names:
        vals = [m[k] for m in all_metrics]
        results[k] = (float(np.mean(vals)), float(np.std(vals)))
    return results 

def compute_map(gt_segments: List[Segment], pred_segments: List[Segment]):
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
                    
                    iou = calculate_iou(gt_seg, pred_seg)
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
        
    return np.mean(ap_scores)

def calculate_iou(seg1: Segment, seg2: Segment) -> float:
    """Calculate Intersection over Union between two segments."""
    intersection_start = max(seg1.start_time, seg2.start_time)
    intersection_end = min(seg1.end_time, seg2.end_time)
    
    if intersection_end <= intersection_start:
        return 0.0
    
    intersection_duration = intersection_end - intersection_start
    union_duration = (seg1.end_time - seg1.start_time) + (seg2.end_time - seg2.start_time) - intersection_duration
    
    return intersection_duration / union_duration if union_duration > 0 else 0.0
