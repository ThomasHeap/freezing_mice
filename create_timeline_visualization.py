#!/usr/bin/env python3
"""
Create timeline visualizations comparing ground truth and predicted annotations.

This script creates timeline plots that show ground truth and predicted behavior 
annotations side-by-side for easy visual comparison. Each video gets its own row
in the plot, with ground truth on the bottom track and predictions on the top track.

Usage:
    python create_timeline_visualization.py --pred-dir /path/to/predictions --gt-dir /path/to/ground_truth
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (20, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

#fixed colours (choose nicer ones like pastels)
behaviour_colors = {
    "grooming": "lightcoral",
    "not_grooming": "red",
    "mount": "lightblue",
    "investigation": "lightgreen",
    "other": "lightgray",
    "attack": "lightblue",
    "scratching": "lightpink",
    "not scratching": "lightgray",
    "not_scratching": "lightgray",
    "bedding box": "lightpink",
    "foraging": "lightgreen",
    "not foraging": "lightgray",
    "Freezing": "lightcoral",
    "Not Freezing": "lightgray",
}

class TimelineVisualizer:
    """Create timeline visualizations comparing ground truth and predictions."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def match_video_files(self, pred_dir: Path, gt_dir: Path, max_videos: int = 5) -> Dict[str, Dict]:
        """Match prediction and ground truth files based on video identifiers."""
        
        # Find prediction files
        pred_files = list(pred_dir.glob("*.json"))
        if not pred_files:
            print(f"❌ No prediction files found in {pred_dir}")
            return {}
        
        # Find ground truth files  
        gt_files = list(gt_dir.glob("*.json"))
        if not gt_files:
            print(f"❌ No ground truth files found in {gt_dir}")
            return {}
        
        print(f"📄 Found {len(pred_files)} prediction files and {len(gt_files)} ground truth files")
        
        # Match prediction and ground truth files
        video_pairs = {}
        
        for pred_file in pred_files:
            pred_name = pred_file.stem
            
            # Extract video identifier from prediction filename
            video_id = self._extract_video_id_from_prediction(pred_name)
            print(f"📝 Extracted video ID '{video_id}' from: {pred_file.name}")
            
            # Find matching ground truth file
            matching_gt = self._find_matching_gt_file(video_id, gt_files)
            
            if matching_gt:
                try:
                    # Load prediction data
                    with open(pred_file, 'r') as f:
                        pred_data = json.load(f)
                    
                    # Load ground truth data
                    with open(matching_gt, 'r') as f:
                        gt_data = json.load(f)
                    
                    # Extract segments
                    pred_segments = self._extract_segments(pred_data, pred_file.name)
                    gt_segments = self._extract_segments(gt_data, matching_gt.name)
                    
                    if pred_segments is not None and gt_segments is not None:
                        video_pairs[video_id] = {
                            'ground_truth': gt_segments,
                            'prediction': pred_segments,
                            'pred_file': pred_file.name,
                            'gt_file': matching_gt.name
                        }
                        print(f"  ✅ Successfully matched with: {matching_gt.name}")
                    
                except Exception as e:
                    print(f"  ❌ Error loading data for {video_id}: {e}")
                    continue
            else:
                print(f"  ⚠️  No matching ground truth found for {video_id}")
        
        print(f"\n🎯 Successfully matched {len(video_pairs)} video pairs")
        
        # Limit number of videos and return
        limited_pairs = dict(list(video_pairs.items())[:max_videos])
        if len(video_pairs) > max_videos:
            print(f"📊 Showing first {max_videos} videos (out of {len(video_pairs)} total)")
        
        return limited_pairs
    
    def _extract_video_id_from_prediction(self, pred_name: str) -> str:
        """Extract video identifier from prediction filename."""
        # New format: {video_id}_gemini-2.5-pro-preview-05-06.json
        if pred_name.endswith('_gemini-2.5-pro-preview-05-06'):
            video_id = pred_name[:-len('_gemini-2.5-pro-preview-05-06')]
            return video_id
        
        # Handle CALMS-style naming: calms_mouse002_task1_annotator1_small_gemini_segments_gemini-2.5-pro-preview-05-06.json
        if pred_name.startswith('calms_'):
            video_id = pred_name[6:]  # Remove 'calms_' prefix
            
            # Remove gemini-specific suffixes
            suffixes_to_remove = [
                '_gemini_segments_gemini-2.5-pro-preview-05-06',
                '_small_gemini_segments_gemini-2.5-pro-preview-05-06',
                '_gemini_segments_gemini-2.0-flash-001',
                '_small_gemini_segments_gemini-2.0-flash-001',
                '_gemini_segments',
                '_small_gemini_segments'
            ]
            
            for suffix in suffixes_to_remove:
                if suffix in video_id:
                    video_id = video_id[:video_id.index(suffix)]
                    break
            
            return video_id
        
        # For other naming patterns, try to remove common prediction suffixes
        suffixes_to_remove = ['_predictions', '_pred', '_gemini', '_segments']
        for suffix in suffixes_to_remove:
            if pred_name.endswith(suffix):
                return pred_name[:-len(suffix)]
        
        # Fallback: return the filename as is
        return pred_name
    
    def _find_matching_gt_file(self, video_id: str, gt_files: List[Path]) -> Optional[Path]:
        """Find the ground truth file that matches the video ID."""
        for gt_file in gt_files:
            gt_stem = gt_file.stem
            
            # Remove common ground truth suffixes
            # Handle patterns like: mouse001_task1_annotator1.annotator-id_0.json
            if '.annotator-id_' in gt_stem:
                gt_video_id = gt_stem[:gt_stem.index('.annotator-id_')]
            else:
                gt_video_id = gt_stem
            
            # Check for exact match
            if video_id == gt_video_id:
                return gt_file
            
            # Check for partial matches (in case of slight naming differences)
            if video_id in gt_video_id or gt_video_id in video_id:
                return gt_file
        
        return None
    
    def _extract_segments(self, data: dict, filename: str) -> Optional[List[dict]]:
        """Extract segments from JSON data."""
        if isinstance(data, dict) and 'segments' in data:
            return data['segments']
        elif isinstance(data, list):
            return data
        else:
            print(f"  ❌ Unexpected format in {filename}")
            return None
    
    def create_timeline_plot(self, video_pairs: Dict[str, Dict], dataset_name: str):
        """Create the timeline visualization plot."""
        if not video_pairs:
            print("❌ No video pairs to plot")
            return
        
        video_names = list(video_pairs.keys())
        n_videos = len(video_names)
        
        # Calculate figure height based on number of videos
        fig_height = max(8, n_videos * 3)
        fig, axes = plt.subplots(n_videos, 1, figsize=(20, fig_height), squeeze=False)
        axes = axes.flatten()
        
        # Get all unique behaviors for consistent coloring
        all_behaviors = set()
        for video_data in video_pairs.values():
            for seg in video_data['ground_truth']:
                behavior = seg.get('behavior', '')
                if behavior and behavior != 'background':
                    all_behaviors.add(behavior)
            for seg in video_data['prediction']:
                behavior = seg.get('behavior', '')
                if behavior and behavior != 'background':
                    all_behaviors.add(behavior)
        
        all_behaviors = sorted(list(all_behaviors))
        
        # Create color map for behaviors
        if all_behaviors:
            behavior_colors = {}
            colors = [behaviour_colors[behavior] for behavior in all_behaviors]
            for behavior, color in zip(all_behaviors, colors):
                behavior_colors[behavior] = color
        else:
            behavior_colors = {}
        
        # Plot each video
        for idx, video_name in enumerate(video_names):
            ax = axes[idx]
            video_data = video_pairs[video_name]
            
            gt_segments = video_data['ground_truth']
            pred_segments = video_data['prediction']
            
            # Normalize segments and plot
            normalized_gt = [self._normalize_segment(seg) for seg in gt_segments]
            normalized_pred = [self._normalize_segment(seg) for seg in pred_segments]
            
            # Find video duration
            max_time = 0
            for start, end, _ in normalized_gt + normalized_pred:
                max_time = max(max_time, end)
            
            if max_time == 0:
                continue
            
            # Plot ground truth (bottom track)
            self._plot_track(ax, normalized_gt, 0, behavior_colors, max_time, "Ground Truth")
            
            # Plot predictions (top track)
            self._plot_track(ax, normalized_pred, 1, behavior_colors, max_time, "Prediction")
            
            # Format the subplot
            ax.set_xlim(0, max_time)
            ax.set_ylim(-0.1, 1.9)
            ax.set_yticks([0.2, 1.2])
            ax.set_yticklabels(['Ground Truth', 'Prediction'])
            ax.set_xlabel('Time (seconds)', fontsize=10)
            ax.set_title(f'{video_name}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add time ticks
            time_ticks = np.arange(0, max_time + 1, max(1, int(max_time / 10)))
            ax.set_xticks(time_ticks)
        
        # Create legend for behaviors
        if all_behaviors:
            legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=behavior_colors[behavior], 
                                           alpha=0.8, edgecolor='black') 
                             for behavior in all_behaviors]
            fig.legend(legend_elements, all_behaviors, 
                      loc='upper center', bbox_to_anchor=(0.5, 0.98), 
                      ncol=min(len(all_behaviors), 6), fontsize=10)
        
        plt.tight_layout()
        
        # Adjust top margin if we have a legend
        if all_behaviors:
            plt.subplots_adjust(top=0.92)
        
        # Save the plot
        output_name = f'video_timeline_{dataset_name}'
        plt.savefig(self.output_dir / f'{output_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / f'{output_name}.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✅ Timeline visualization saved:")
        print(f"   📄 {self.output_dir / f'{output_name}.png'}")
        print(f"   📄 {self.output_dir / f'{output_name}.pdf'}")
    
    def _normalize_segment(self, seg: dict) -> Tuple[float, float, str]:
        """Normalize segment to (start_time, end_time, behavior) format."""
        if isinstance(seg, dict):
            start_time = seg.get('start_time', 0)
            end_time = seg.get('end_time', 0)
            behavior = seg.get('behavior', 'unknown')
        else:
            start_time = getattr(seg, 'start_time', 0)
            end_time = getattr(seg, 'end_time', 0)
            behavior = getattr(seg, 'behavior', 'unknown')
        
        # Convert time format if needed (MM:SS to seconds)
        if isinstance(start_time, str) and ':' in start_time:
            parts = start_time.split(':')
            start_time = float(parts[0]) * 60 + float(parts[1])
        if isinstance(end_time, str) and ':' in end_time:
            parts = end_time.split(':')
            end_time = float(parts[0]) * 60 + float(parts[1])
        
        return float(start_time), float(end_time), behavior
    
    def _plot_track(self, ax, segments: List[Tuple], y_pos: int, 
                   behavior_colors: Dict[str, str], max_time: float, track_name: str):
        """Plot a single track (ground truth or prediction)."""
        for start_time, end_time, behavior in segments:
            if behavior == 'background':
                continue
                
            duration = end_time - start_time
            color = behavior_colors.get(behavior, 'gray')
            
            # Draw segment rectangle
            rect = plt.Rectangle((start_time, y_pos), duration, 0.4, 
                               facecolor=color, alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add behavior label if segment is wide enough
            if duration > max_time * 0.02:  # Only label if segment is >2% of video
                ax.text(start_time + duration/2, y_pos + 0.2, behavior, 
                       ha='center', va='center', fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7))
    
    def generate_timeline(self, pred_dir: Path, gt_dir: Path, dataset_name: str, 
                         max_videos: int = 5) -> bool:
        """Main method to generate timeline visualization."""
        print(f"📺 Creating timeline visualization for {dataset_name}")
        print(f"📁 Predictions: {pred_dir}")
        print(f"📁 Ground truth: {gt_dir}")
        print(f"📁 Output: {self.output_dir}")
        
        # Match video files
        video_pairs = self.match_video_files(pred_dir, gt_dir, max_videos)
        
        if not video_pairs:
            print("❌ No matching video pairs found")
            return False
        
        # Create timeline plot
        self.create_timeline_plot(video_pairs, dataset_name)
        
        # Print summary
        print(f"\n📊 Timeline Summary:")
        for video_id, data in video_pairs.items():
            print(f"  🎬 {video_id}:")
            print(f"     GT: {len(data['ground_truth'])} segments ({data['gt_file']})")
            print(f"     Pred: {len(data['prediction'])} segments ({data['pred_file']})")
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description='Create timeline visualizations comparing ground truth and predicted annotations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python create_timeline_visualization.py --pred-dir ./predictions --gt-dir ./ground_truth
  
  # With custom output directory and dataset name
  python create_timeline_visualization.py \\
      --pred-dir /path/to/predictions \\
      --gt-dir /path/to/ground_truth \\
      --output-dir ./timeline_plots \\
      --dataset-name my_dataset \\
      --max-videos 10
      
  # CALMS dataset example
  python create_timeline_visualization.py \\
      --pred-dir ./gemini_mouse_behavior_results/calms \\
      --gt-dir ./data/calms21/per_video_annot_segment \\
      --dataset-name calms \\
      --max-videos 5
        """
    )
    
    parser.add_argument('--pred-dir', required=True, type=Path,
                       help='Directory containing prediction JSON files')
    parser.add_argument('--gt-dir', required=True, type=Path,
                       help='Directory containing ground truth JSON files')
    parser.add_argument('--output-dir', default='./timeline_visualizations', type=Path,
                       help='Directory to save timeline plots (default: ./timeline_visualizations)')
    parser.add_argument('--dataset-name', default='dataset',
                       help='Name for the dataset (used in output filenames, default: dataset)')
    parser.add_argument('--max-videos', type=int, default=5,
                       help='Maximum number of videos to include in timeline (default: 5)')
    
    args = parser.parse_args()
    
    # Validate input directories
    if not args.pred_dir.exists():
        print(f"❌ Prediction directory not found: {args.pred_dir}")
        return 1
    
    if not args.gt_dir.exists():
        print(f"❌ Ground truth directory not found: {args.gt_dir}")
        return 1
    
    # Create visualizer and generate timeline
    visualizer = TimelineVisualizer(args.output_dir)
    
    success = visualizer.generate_timeline(
        args.pred_dir, 
        args.gt_dir, 
        args.dataset_name, 
        args.max_videos
    )
    
    if success:
        print(f"\n🎉 Timeline visualization complete!")
        return 0
    else:
        print(f"\n❌ Timeline visualization failed!")
        return 1

if __name__ == "__main__":
    exit(main()) 