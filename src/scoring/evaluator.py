def time_to_seconds(time_str):
    """Convert MM:SS format to seconds"""
    minutes, seconds = map(float, time_str.split(":"))
    return minutes * 60 + seconds

def seconds_to_time(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

def get_behavior_at_second(segments, second, window_size=2):
    """
    Get the behavior occurring at a specific second, considering a window around it.
    Returns a set of behaviors that occur within the window.
    """
    behaviors = set()
    for seg in segments:
        #if segment is not dict, convert to dict
        if isinstance(seg, dict):
            seg = seg
        else:
            seg = {"start_time": seg.start_time, "end_time": seg.end_time, "behavior": seg.behavior}
        start_sec = time_to_seconds(seg["start_time"])
        end_sec = time_to_seconds(seg["end_time"])
        
        # Check if the second falls within the window of this segment
        if (start_sec - window_size <= second <= end_sec + window_size):
            behaviors.add(seg["behavior"].lower())
    return behaviors

def score_predictions(ground_truth, predictions):
    """
    Score predictions on a per-second basis with a 2-second window for matching.
    
    Args:
        ground_truth: List of ground truth segments
        predictions: List of predicted segments
    
    Returns:
        Dictionary containing per-second metrics and overall statistics
    """
    # Convert ground truth to list if it's not already
    if isinstance(ground_truth, dict) and "segments" in ground_truth:
        ground_truth = ground_truth["segments"]
    
    # Find the total duration of the video
    max_time = 0
    for seg in ground_truth:
        end_sec = time_to_seconds(seg["end_time"])
        max_time = max(max_time, end_sec)
    
    # Initialize counters
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    per_second_results = []
    
    # Evaluate each second
    for second in range(int(max_time) + 1):
        # Get behaviors in ground truth and predictions for this second
        gt_behaviors = get_behavior_at_second(ground_truth, second)
        pred_behaviors = get_behavior_at_second(predictions, second)
        
        # For each behavior type that appears in either ground truth or predictions
        all_behaviors = gt_behaviors.union(pred_behaviors)
        
        for behavior in all_behaviors:
            # Check if behavior is present in ground truth
            gt_has_behavior = behavior in gt_behaviors
            pred_has_behavior = behavior in pred_behaviors
            
            if gt_has_behavior and pred_has_behavior:
                true_positives += 1
            elif not gt_has_behavior and not pred_has_behavior:
                true_negatives += 1
            elif gt_has_behavior and not pred_has_behavior:
                false_negatives += 1
            elif not gt_has_behavior and pred_has_behavior:
                false_positives += 1
            
            # Record per-second result
            per_second_results.append({
                "second": second,
                "behavior": behavior,
                "ground_truth": gt_has_behavior,
                "prediction": pred_has_behavior,
                "is_correct": gt_has_behavior == pred_has_behavior
            })
    
    # Calculate metrics
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "per_second_results": per_second_results,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": true_positives,
            "true_negatives": true_negatives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    } 