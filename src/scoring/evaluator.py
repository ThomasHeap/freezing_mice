def calculate_segment_overlap(gt_start, gt_end, pred_start, pred_end):
    """Calculate the overlap between two time segments in seconds"""
    gt_start_sec = gt_start
    gt_end_sec = gt_end
    pred_start_sec = pred_start
    pred_end_sec = pred_end
    
    overlap_start = max(gt_start_sec, pred_start_sec)
    overlap_end = min(gt_end_sec, pred_end_sec)
    
    return max(0, overlap_end - overlap_start)

def score_predictions(ground_truth, predictions, overlap_threshold=0.5):
    """
    Score the predictions against ground truth annotations
    
    Args:
        ground_truth: List of ground truth segments
        predictions: List of predicted segments
        overlap_threshold: Minimum overlap ratio to consider segments matching
    
    Returns:
        Dictionary containing precision, recall, F1 score, and overlap statistics
    """
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    overlap_percentages = []  # Track overlap percentages for true positives
    
    # Convert ground truth to list if it's not already
    if isinstance(ground_truth, dict) and "segments" in ground_truth:
        ground_truth = ground_truth["segments"]
    
    # For each ground truth segment
    for gt_seg in ground_truth:
        gt_duration = gt_seg["end_time"] - gt_seg["start_time"]
        matched = False
        
        # Check against each prediction
        for pred_seg in predictions:
            # Calculate overlap
            overlap = calculate_segment_overlap(
                gt_seg["start_time"], gt_seg["end_time"],
                pred_seg.start_time, pred_seg.end_time
            )
            
            if gt_duration == 0:
                continue
            
            overlap_ratio = overlap / gt_duration
            
            # If significant overlap and same behavior
            if (overlap_ratio >= overlap_threshold and 
                gt_seg["behavior"].lower() == pred_seg.behavior.lower()):
                true_positives += 1
                matched = True
                overlap_percentages.append(overlap_ratio * 100)  # Convert to percentage
                break
        
        if not matched:
            false_negatives += 1
    
    # Count false positives (predictions that don't match any ground truth)
    for pred_seg in predictions:
        pred_duration = pred_seg.end_time - pred_seg.start_time
        matched = False
        
        for gt_seg in ground_truth:
            overlap = calculate_segment_overlap(
                gt_seg["start_time"], gt_seg["end_time"],
                pred_seg.start_time, pred_seg.end_time
            )
            
            if pred_duration == 0:
                continue
            
            if ((overlap / pred_duration) >= overlap_threshold and 
                gt_seg["behavior"].lower() == pred_seg.behavior.lower()):
                matched = True
                break
        
        if not matched:
            false_positives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate overlap statistics
    avg_overlap = sum(overlap_percentages) / len(overlap_percentages) if overlap_percentages else 0
    min_overlap = min(overlap_percentages) if overlap_percentages else 0
    max_overlap = max(overlap_percentages) if overlap_percentages else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "overlap_stats": {
            "average_overlap_percentage": avg_overlap,
            "min_overlap_percentage": min_overlap,
            "max_overlap_percentage": max_overlap,
            "all_overlap_percentages": overlap_percentages
        }
    } 