import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import pandas as pd
from dataclasses import dataclass

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class ComparativeVisualizer:
    """Create comparative visualizations across multiple datasets/models."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_results(self, results_dir: Path) -> Dict[str, Dict]:
        """Collect all evaluation results from subdirectories."""
        results = {}
        
        # Look for JSON files in subdirectories
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                # Look specifically for evaluation_results.json files
                json_files = list(subdir.glob("evaluation_results.json"))
                
                for json_file in json_files:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        
                        # Use the subfolder name as the dataset name
                        dataset_name = subdir.name
                        
                        results[dataset_name] = data
                        print(f"✅ Loaded results from: {json_file}")
                        
                    except Exception as e:
                        print(f"❌ Error loading {json_file}: {e}")
        
        return results
    
    def plot_core_metrics_comparison(self, results: Dict[str, Dict]):
        """Plot comparison of core metrics across datasets."""
        datasets = list(results.keys())
        metrics = ['second_accuracy', 'mAP', 'edit_distance']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Second-wise accuracy
        accuracies = [results[ds].get('second_accuracy', 0) for ds in datasets]
        bars1 = axes[0].bar(datasets, accuracies, color=sns.color_palette("viridis", len(datasets)))
        axes[0].set_title('Second-wise Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # mAP
        maps = [results[ds].get('mAP', 0) for ds in datasets]
        bars2 = axes[1].bar(datasets, maps, color=sns.color_palette("plasma", len(datasets)))
        axes[1].set_title('mean Average Precision (mAP)', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('mAP')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        for bar, map_val in zip(bars2, maps):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{map_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Edit Distance (lower is better)
        edit_dists = [results[ds].get('edit_distance', 0) for ds in datasets]
        bars3 = axes[2].bar(datasets, edit_dists, color=sns.color_palette("rocket", len(datasets)))
        axes[2].set_title('Edit Distance (Lower = Better)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Edit Distance')
        axes[2].tick_params(axis='x', rotation=45)
        
        for bar, ed in zip(bars3, edit_dists):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(edit_dists)*0.01, 
                        f'{ed}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'core_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'core_metrics_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_boundary_tolerance_analysis(self, results: Dict[str, Dict]):
        """Plot boundary detection performance across tolerances for all datasets."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        tolerances = [0.5, 1.0, 2.0, 3.0]
        
        for dataset, data in results.items():
            boundary_data = data.get('boundary_tolerance_analysis', {})
            
            precisions = []
            recalls = []
            f1s = []
            
            for tol in tolerances:
                tol_key = f"tolerance_{tol}s"
                if tol_key in boundary_data:
                    precisions.append(boundary_data[tol_key]['precision'])
                    recalls.append(boundary_data[tol_key]['recall'])
                    f1s.append(boundary_data[tol_key]['f1'])
                else:
                    precisions.append(0)
                    recalls.append(0)
                    f1s.append(0)
            
            ax1.plot(tolerances, precisions, marker='o', linewidth=2, markersize=8, label=dataset)
            ax2.plot(tolerances, recalls, marker='s', linewidth=2, markersize=8, label=dataset)
            ax3.plot(tolerances, f1s, marker='^', linewidth=2, markersize=8, label=dataset)
        
        # Customize plots
        for i, (ax, metric) in enumerate(zip([ax1, ax2, ax3], ['Precision', 'Recall', 'F1-Score'])):
            ax.set_xlabel('Temporal Tolerance (seconds)', fontsize=12)
            ax.set_ylabel(f'Boundary {metric}', fontsize=12)
            ax.set_title(f'Boundary {metric} vs Tolerance', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            # Only show legend on the rightmost plot
            if i == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'boundary_tolerance_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'boundary_tolerance_analysis.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_iou_threshold_analysis(self, results: Dict[str, Dict]):
        """Plot IoU-based metrics across different thresholds."""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        
        iou_thresholds = [0.3, 0.5, 0.7]
        
        for dataset, data in results.items():
            precisions = []
            recalls = []
            f1s = []
            
            for threshold in iou_thresholds:
                precisions.append(data.get(f'iou_{threshold}_precision', 0))
                recalls.append(data.get(f'iou_{threshold}_recall', 0))
                f1s.append(data.get(f'iou_{threshold}_f1', 0))
            
            ax1.plot(iou_thresholds, precisions, marker='o', linewidth=3, markersize=10, label=dataset)
            ax2.plot(iou_thresholds, recalls, marker='s', linewidth=3, markersize=10, label=dataset)
            ax3.plot(iou_thresholds, f1s, marker='^', linewidth=3, markersize=10, label=dataset)
        
        # Customize plots
        for i, (ax, metric) in enumerate(zip([ax1, ax2, ax3], ['Precision', 'Recall', 'F1-Score'])):
            ax.set_xlabel('IoU Threshold', fontsize=12)
            ax.set_ylabel(f'Segment {metric}', fontsize=12)
            ax.set_title(f'Segment {metric} vs IoU Threshold', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            # Only show legend on the rightmost plot
            if i == 2:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'iou_threshold_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'iou_threshold_analysis.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_per_class_performance(self, results: Dict[str, Dict]):
        """Plot per-class performance comparison."""
        # Collect all unique behaviors across datasets
        all_behaviors = set()
        for data in results.values():
            if 'per_class_metrics' in data:
                all_behaviors.update(data['per_class_metrics'].keys())
        
        all_behaviors = sorted(list(all_behaviors))
        
        if not all_behaviors:
            print("No per-class metrics found for visualization")
            return
        
        # Create subplots for each metric
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['precision', 'recall', 'f1', 'support']
        metric_titles = ['Precision', 'Recall', 'F1-Score', 'Support (# instances)']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx]
            
            # Prepare data for grouped bar chart
            x = np.arange(len(all_behaviors))
            width = 0.8 / len(results)
            
            for i, (dataset, data) in enumerate(results.items()):
                values = []
                for behavior in all_behaviors:
                    class_data = data.get('per_class_metrics', {}).get(behavior, {})
                    values.append(class_data.get(metric, 0))
                
                offset = (i - len(results)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, label=dataset, alpha=0.8)
                
                # Add value labels for small datasets
                if len(results) <= 3 and metric != 'support':
                    for bar, val in zip(bars, values):
                        if val > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                                   f'{val:.2f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel('Behavior Class', fontsize=12)
            ax.set_ylabel(title, fontsize=12)
            ax.set_title(f'Per-Class {title}', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(all_behaviors, rotation=45, ha='right')
            
            if metric != 'support':
                ax.set_ylim(0, 1)
            
            # Only show legend on the last subplot
            if idx == len(metrics) - 1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'per_class_performance.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_duration_analysis(self, results: Dict[str, Dict]):
        """Plot duration analysis comparison."""
        # Collect all behaviors
        all_behaviors = set()
        for data in results.values():
            if 'duration_analysis' in data:
                all_behaviors.update(data['duration_analysis'].keys())
        
        all_behaviors = sorted(list(all_behaviors))
        
        if not all_behaviors:
            print("No duration analysis found for visualization")
            return
        
        # Create plots for duration bias and mean durations
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        
        # Duration Bias Plot
        x = np.arange(len(all_behaviors))
        width = 0.8 / len(results)
        
        for i, (dataset, data) in enumerate(results.items()):
            biases = []
            for behavior in all_behaviors:
                duration_data = data.get('duration_analysis', {}).get(behavior, {})
                biases.append(duration_data.get('duration_bias', 0))
            
            offset = (i - len(results)/2 + 0.5) * width
            bars = ax1.bar(x + offset, biases, width, label=dataset, alpha=0.8)
            
            # Add value labels
            for bar, bias in zip(bars, biases):
                if abs(bias) > 0.1:  # Only label significant biases
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.1 if bias > 0 else -0.1), 
                            f'{bias:+.1f}s', ha='center', va='bottom' if bias > 0 else 'top', fontsize=8)
        
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Behavior Class', fontsize=12)
        ax1.set_ylabel('Duration Bias (seconds)', fontsize=12)
        ax1.set_title('Duration Bias by Behavior (Predicted - Ground Truth)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_behaviors, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Mean Duration Comparison
        for i, (dataset, data) in enumerate(results.items()):
            gt_means = []
            pred_means = []
            
            for behavior in all_behaviors:
                duration_data = data.get('duration_analysis', {}).get(behavior, {})
                gt_means.append(duration_data.get('gt_mean', 0))
                pred_means.append(duration_data.get('pred_mean', 0))
            
            offset = (i - len(results)/2 + 0.5) * width
            ax2.bar(x + offset - width/4, gt_means, width/2, label=f'{dataset} (GT)', alpha=0.6)
            ax2.bar(x + offset + width/4, pred_means, width/2, label=f'{dataset} (Pred)', alpha=0.8)
        
        ax2.set_xlabel('Behavior Class', fontsize=12)
        ax2.set_ylabel('Mean Duration (seconds)', fontsize=12)
        ax2.set_title('Mean Duration Comparison: Ground Truth vs Predictions', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(all_behaviors, rotation=45, ha='right')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'duration_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'duration_analysis.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_segmentation_quality(self, results: Dict[str, Dict]):
        """Plot segmentation quality metrics."""
        datasets = list(results.keys())
        
        over_seg_errors = [results[ds].get('over_segmentation_error', 0) for ds in datasets]
        under_seg_errors = [results[ds].get('under_segmentation_error', 0) for ds in datasets]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Over-segmentation errors
        bars1 = ax1.bar(datasets, over_seg_errors, color=sns.color_palette("Reds_r", len(datasets)), alpha=0.8)
        ax1.set_title('Over-segmentation Error', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Error Rate')
        ax1.set_ylim(0, max(max(over_seg_errors), max(under_seg_errors)) * 1.1)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, error in zip(bars1, over_seg_errors):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{error:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Under-segmentation errors
        bars2 = ax2.bar(datasets, under_seg_errors, color=sns.color_palette("Blues_r", len(datasets)), alpha=0.8)
        ax2.set_title('Under-segmentation Error', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Error Rate')
        ax2.set_ylim(0, max(max(over_seg_errors), max(under_seg_errors)) * 1.1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, error in zip(bars2, under_seg_errors):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{error:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'segmentation_quality.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'segmentation_quality.pdf', bbox_inches='tight')
        plt.close()
    
    def create_summary_table(self, results: Dict[str, Dict]):
        """Create a comprehensive summary table."""
        summary_data = []
        
        for dataset, data in results.items():
            row = {
                'Dataset': dataset,
                'Second Accuracy': f"{data.get('second_accuracy', 0):.4f}",
                'mAP': f"{data.get('mAP', 0):.4f}",
                'Edit Distance': data.get('edit_distance', 0),
                'Boundary F1 (±1s)': f"{data.get('boundary_f1', 0):.4f}",
                'IoU@0.5 F1': f"{data.get('iou_0.5_f1', 0):.4f}",
                'Over-seg Error': f"{data.get('over_segmentation_error', 0):.4f}",
                'Under-seg Error': f"{data.get('under_segmentation_error', 0):.4f}",
                'Total Seconds': data.get('total_seconds', 0)
            }
            summary_data.append(row)
        
        # Create table plot
        fig, ax = plt.subplots(figsize=(16, max(6, len(summary_data) * 0.5)))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        df = pd.DataFrame(summary_data)
        table = ax.table(cellText=df.values, colLabels=df.columns, 
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Color header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Evaluation Summary Table', fontsize=16, fontweight='bold', pad=20)
        plt.savefig(self.output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'summary_table.pdf', bbox_inches='tight')
        plt.close()
        
        # Also save as CSV
        df.to_csv(self.output_dir / 'summary_table.csv', index=False)
    
    def generate_all_plots(self, results_dir: Path):
        """Generate all comparative visualizations."""
        print(f"🔍 Scanning for results in: {results_dir}")
        results = self.collect_results(results_dir)
        
        if not results:
            print("❌ No evaluation results found!")
            return
        
        print(f"📊 Found {len(results)} datasets: {list(results.keys())}")
        print(f"📈 Generating visualizations in: {self.output_dir}")
        
        # Generate all plots
        self.plot_core_metrics_comparison(results)
        print("✅ Core metrics comparison")
        
        self.plot_boundary_tolerance_analysis(results)
        print("✅ Boundary tolerance analysis")
        
        self.plot_iou_threshold_analysis(results)
        print("✅ IoU threshold analysis")
        
        self.plot_per_class_performance(results)
        print("✅ Per-class performance")
        
        self.plot_duration_analysis(results)
        print("✅ Duration analysis")
        
        self.plot_segmentation_quality(results)
        print("✅ Segmentation quality")
        
        self.create_summary_table(results)
        print("✅ Summary table")
        
        print(f"\n🎉 All visualizations saved to: {self.output_dir}")
        print("📁 Generated files:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"   - {file.name}")

def main():
    parser = argparse.ArgumentParser(description='Generate comparative visualizations for action segmentation evaluation')
    parser.add_argument('--results-dir', required=True, 
                       help='Directory containing subdirectories with evaluation results')
    parser.add_argument('--output-dir', default='./comparative_plots',
                       help='Directory to save plots (default: ./comparative_plots)')
    
    args = parser.parse_args()
    
    # Create visualizer and generate plots
    visualizer = ComparativeVisualizer(args.output_dir)
    visualizer.generate_all_plots(Path(args.results_dir))

if __name__ == "__main__":
    main()