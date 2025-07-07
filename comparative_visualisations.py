import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
from matplotlib.patches import Patch
# No longer using renewal baseline - using predicted segment proportions instead

plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 15
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

behaviour_colors = {
    "grooming": "lightcoral",
    "not_grooming": "red",
    "mount": "lightblue",
    "investigation": "lightgreen",
    "other": "lightgray",
    "attack": "lightblue",
    "scratching": "lightpink",
    "not scratching": "lightgray",
    "bedding box": "lightpink",
    "Freezing": "lightcoral",
    "Not Freezing": "lightgray",
    "foraging": "lightgreen",
    "not foraging": "lightgray",
    "background": "lightgray",
    "scratch": "lightpink",
    "lick": "lightblue",
}

def parse_time_str(time_str):
    """Parse a time string in 'M:SS' or 'MM:SS' format to seconds (int)."""
    if isinstance(time_str, (int, float)):
        return float(time_str)
    if not isinstance(time_str, str):
        return 0.0
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + int(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
    else:
        try:
            return float(time_str)
        except Exception:
            return 0.0

class ComparativeVisualizer:
    """Create comparative visualizations across multiple datasets/models (core metrics only)."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_colors = {}
    
    def get_dataset_colors(self, datasets: List[str]) -> Dict[str, str]:
        if self.dataset_colors == {}:
            colors = sns.color_palette("husl", len(datasets))
            self.dataset_colors = {dataset: colors[i] for i, dataset in enumerate(sorted(datasets))}
        return self.dataset_colors
    
    def get_dataset_color(self, dataset: str) -> str:
        return self.dataset_colors[dataset]
    
    def collect_results(self, results_dir: Path) -> Dict[str, Dict]:
        results = {}
        for subdir in results_dir.iterdir():
            if subdir.is_dir():
                dataset_name = subdir.name
                
                # Look for aggregated_results.json files first, then fall back to evaluation_results.json
                aggregated_file = subdir / "aggregated_results.json"
                evaluation_file = subdir / "evaluation_results.json"
                
                # Prioritize aggregated results over evaluation results
                json_file = None
                if aggregated_file.exists():
                    json_file = aggregated_file
                elif evaluation_file.exists():
                    json_file = evaluation_file
                
                if json_file:
                    try:
                        with open(json_file, 'r') as f:
                            data = json.load(f)
                        results[dataset_name] = data
                        print(f"✅ Loaded results from: {json_file}")
                    except Exception as e:
                        print(f"❌ Error loading {json_file}: {e}")
        return results
    
    def calculate_chance_performance(self, results: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate theoretically sound chance performance for all metrics, using predicted segment proportions."""
        chance_metrics = {}
        for dataset, data in results.items():
            chance_metrics[dataset] = {}
            
            # Check if we have aggregated results with files data
            if 'files' in data:
                # Calculate baselines using predicted segment proportions from all files
                total_pred_durations = {}
                total_video_duration = 0
                
                # Aggregate predicted durations across all files
                for filename, file_data in data['files'].items():
                    if 'temporal_coverage' in file_data:
                        total_video_duration += file_data.get('total_seconds', 0)
                        
                        for behavior, coverage in file_data['temporal_coverage'].items():
                            if behavior not in total_pred_durations:
                                total_pred_durations[behavior] = 0
                            total_pred_durations[behavior] += coverage.get('pred_duration', 0)
                
                # Calculate class proportions from predicted segments
                if total_pred_durations and total_video_duration > 0:
                    class_proportions = []
                    for behavior, pred_duration in total_pred_durations.items():
                        if pred_duration > 0:
                            proportion = pred_duration / total_video_duration
                            class_proportions.append(proportion)
                    
                    if class_proportions:
                        # Calculate chance performance metrics based on predicted proportions
                        chance_accuracy = sum(p * p for p in class_proportions)
                        chance_macro_f1 = np.mean(class_proportions)
                        chance_mAP = 0
                        
                        entropy = 0
                        
                        # For MCC, use a simple baseline based on class balance
                        # MCC baseline is typically around 0 for random classification
                        chance_mcc = 0.0
                        
                        chance_metrics[dataset] = {
                            'second_accuracy': chance_accuracy,
                            'macro_f1': chance_macro_f1,
                            'mAP': chance_mAP,
                            'mutual_info_gt_vs_pred': entropy,
                            'MCC': chance_mcc
                        }
                        
                        print(f"✅ Calculated baselines for {dataset} using predicted proportions")
                        print(f"   Predicted class proportions: {class_proportions}")
                        print(f"   Chance accuracy: {chance_accuracy:.3f}")
                        print(f"   Chance macro F1: {chance_macro_f1:.3f}")
                        print(f"   Chance mAP: {chance_mAP:.3f}")
                        print(f"   Predicted entropy: {entropy:.3f}")
                    else:
                        print(f"⚠️  No valid predicted proportions found for {dataset}")
                        chance_metrics[dataset] = {
                            'second_accuracy': 0.0,
                            'macro_f1': 0.0,
                            'mAP': 0.0,
                            'mutual_info_gt_vs_pred': 0.0,
                            'MCC': 0.0
                        }
                else:
                    print(f"⚠️  No temporal coverage data found for {dataset}")
                    chance_metrics[dataset] = {
                        'second_accuracy': 0.0,
                        'macro_f1': 0.0,
                        'mAP': 0.0,
                        'mutual_info_gt_vs_pred': 0.0,
                        'MCC': 0.0
                    }
            else:
                # Fallback for non-aggregated results
                print(f"⚠️  No files data found for {dataset}, using fallback baselines")
                chance_metrics[dataset] = {
                    'second_accuracy': 0.25,  # Simple fallback for binary classification
                    'macro_f1': 0.25,
                    'mAP': 0.25,
                    'mutual_info_gt_vs_pred': 0.5,
                    'MCC': 0.0
                }
                    
        return chance_metrics
    
    def load_user_performance(self, my_version_dir: Path = Path("results/my_version")) -> Dict[str, Dict]:
        # Return empty dict - no longer loading user performance
        return {}
    
    def calculate_maximum_values(self, results: Dict[str, Dict]) -> Dict[str, Dict[str, float]]:
        """Calculate maximum possible values for mutual information and MCC for each dataset."""
        max_values = {}
        
        for dataset, data in results.items():
            max_values[dataset] = {}
            
            # For aggregated results, calculate from individual files
            if 'files' in data:
                # Calculate maximum mutual information per file
                file_max_mi = []
                
                for filename, file_data in data['files'].items():
                    if 'temporal_coverage' in file_data:
                        # Extract ground truth labels from this file's temporal coverage
                        file_labels = []
                        
                        for behavior, coverage in file_data['temporal_coverage'].items():
                            gt_duration = coverage.get('gt_duration', 0)
                            if gt_duration > 0:
                                # Add labels for each second (simplified approximation)
                                file_labels.extend([behavior] * int(gt_duration))
                        
                        if file_labels:
                            # Calculate entropy of this file's ground truth labels
                            from collections import Counter
                            label_counts = Counter(file_labels)
                            total_labels = len(file_labels)
                            
                            # Calculate entropy H(X) for this file
                            entropy = 0
                            for count in label_counts.values():
                                p = count / total_labels
                                if p > 0:
                                    entropy -= p * np.log2(p)
                            
                            file_max_mi.append(entropy)
                        else:
                            file_max_mi.append(1.0)  # Default max for binary classification
                    else:
                        file_max_mi.append(1.0)  # Default max
                
                # Use the maximum entropy across all files as the dataset maximum
                if file_max_mi:
                    max_values[dataset]['mutual_info_gt_vs_pred'] = max(file_max_mi)
                else:
                    max_values[dataset]['mutual_info_gt_vs_pred'] = 1.0
                
                # Maximum MCC is always 1.0 (perfect correlation)
                max_values[dataset]['MCC'] = 1.0
            else:
                # Fallback for non-aggregated results
                max_values[dataset]['mutual_info_gt_vs_pred'] = 1.0
                max_values[dataset]['MCC'] = 1.0
        
        return max_values
    
    def plot_core_metrics_comparison(self, results: Dict[str, Dict], output_dir: Path, plot_title: str = None):
        datasets = list(results.keys())
        print(f"  Plotting datasets: {datasets}")
        self.get_dataset_colors(datasets)
        chance_metrics = self.calculate_chance_performance(results)
        max_values = self.calculate_maximum_values(results)
        
        metrics = [
            ('second_accuracy', 'Second-wise Accuracy', 0, 1),
            ('macro_f1', 'Macro F1 (Unweighted)', 0, 1),
            ('mAP', 'mean Average Precision (mAP)', 0, 1),
            ('mutual_info_gt_vs_pred', 'Mutual Info (GT vs Pred)', 0, 1),
            ('MCC', 'Matthew\'s Correlation Coefficient (MCC)', -1, 1),
        ]
        
        # Create figure for aggregate plots
        fig, axes = plt.subplots(1, 5, figsize=(28, 12))
        axes = axes.flatten()
        
        for idx, (metric_key, title, y_min, y_max) in enumerate(metrics):
            ax = axes[idx]
            
            # Use pre-calculated means and std_errors from aggregated results
            values = []
            error_bars = []
            
            # Check if we have aggregated results with summary statistics
            if 'summary' in results[datasets[0]] and any('_mean' in key for key in results[datasets[0]]['summary'].keys()):
                # Use pre-calculated aggregated results
                for dataset in datasets:
                    mean_key = f'{metric_key}_mean'
                    std_error_key = f'{metric_key}_std_error'
                    if mean_key in results[dataset]['summary']:
                        values.append(results[dataset]['summary'][mean_key])
                        error_bars.append(results[dataset]['summary'].get(std_error_key, 0))
                    else:
                        values.append(0)
                        error_bars.append(0)
            else:
                # Fallback: calculate from individual results
                dataset_groups = {}
                for filename, file_results in results.items():
                    dataset_name = filename.split('_')[0] if '_' in filename else filename
                    if dataset_name not in dataset_groups:
                        dataset_groups[dataset_name] = []
                    dataset_groups[dataset_name].append(file_results)
                
                for dataset in sorted(dataset_groups.keys()):
                    metric_values = []
                    for file_results in dataset_groups[dataset]:
                        if metric_key in file_results:
                            metric_values.append(file_results[metric_key])
                    
                    if metric_values:
                        values.append(np.mean(metric_values))
                        error_bars.append(np.std(metric_values) / np.sqrt(len(metric_values)-1) if len(metric_values) > 1 else 0)
                    else:
                        values.append(0)
                        error_bars.append(0)
            
            colors = [self.get_dataset_color(ds) for ds in datasets]
            bars = ax.bar(range(len(datasets)), values, color=colors, alpha=0.8, 
                         yerr=error_bars, capsize=5)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel(metric_key.replace('_', ' ').title())
            ax.set_xticks([])
            if y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            elif y_min is not None:
                ax.set_ylim(bottom=y_min)
            
            # Add chance performance lines and maximum value lines
            if 'boundary' not in metric_key.lower():
                chance_values = [chance_metrics.get(ds, {}).get(metric_key, 0) for ds in datasets]
                max_values_list = [max_values.get(ds, {}).get(metric_key, 0) for ds in datasets]
                
                for i, (bar, chance_val, max_val) in enumerate(zip(bars, chance_values, max_values_list)):
                    # Add chance performance line
                    if not metric_key.startswith('boundary'):
                        ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], 
                               [chance_val, chance_val], 
                               color='red', linewidth=3, alpha=0.9,
                               label='Chance Performance' if i == 0 and idx == 0 else None)
                    # Add maximum value line for mutual info and MCC
                    if metric_key == 'mutual_info_gt_vs_pred' and max_val > 0:
                        ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], 
                               [max_val, max_val], 
                               color='green', linewidth=3, alpha=0.8, linestyle='--',
                               label='Max Possible' if i == 0 and idx == 3 else None)
                    elif metric_key == 'MCC':
                        ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], 
                               [1.0, 1.0], 
                               color='green', linewidth=3, alpha=0.8, linestyle='--',
                               label='Max Possible' if i == 0 and idx == 3 else None)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                label = f'{val:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(values) * 0.01 if values else 0.01), 
                       label, ha='center', va='bottom', fontweight='bold')
        
        # Hide any extra subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        # Create legend - collect handles from all subplots
        dataset_legend_handles = [Patch(facecolor=self.get_dataset_color(ds), label=ds) for ds in datasets]
        
        # Collect all handles and labels from all subplots
        all_handles = []
        all_labels = []
        seen_labels = set()
        
        for ax in axes:
            if ax.get_visible():  # Only process visible subplots
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label not in seen_labels:
                        all_handles.append(handle)
                        all_labels.append(label)
                        seen_labels.add(label)
        
        # Combine dataset handles with other handles
        all_handles = dataset_legend_handles + all_handles
        all_labels = datasets + all_labels
        
        if all_handles:
            # Place legend below in center as single row
            fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                      fontsize=16, ncol=len(all_handles))
        
        if plot_title:
            fig.suptitle(plot_title, fontsize=24, fontweight='bold')
        
        # Adjust layout to accommodate legend below
        plt.tight_layout(rect=[0, 0.08, 1, 0.97] if plot_title else [0, 0.08, 1, 1])
        
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'core_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'core_metrics_comparison.pdf', bbox_inches='tight')
        plt.close()

    def generate_all_plots(self, results_dir: Path):
        print(f"🔍 Scanning for results in: {results_dir}")
        results = self.collect_results(results_dir)
        if not results:
            print("❌ No evaluation results found!")
            return
        print(f"📊 Found {len(results)} datasets: {list(results.keys())}")
        
        # Debug: print what we found
        for dataset_name, dataset_data in results.items():
            print(f"  Dataset: {dataset_name}")
            if 'summary' in dataset_data:
                print(f"    Has summary with keys: {list(dataset_data['summary'].keys())[:5]}...")
            else:
                print(f"    No summary found")
        
        print(f"📈 Generating visualizations in: {self.output_dir}")
        
        # Create a comparison plot across all datasets
        if any('summary' in data for data in results.values()):
            print(f"📊 Creating comparison plot across all datasets")
            
            # Create the comparison plot
            comparison_dir = self.output_dir / 'comparison'
            self.plot_core_metrics_comparison(
                results, 
                comparison_dir, 
                plot_title='Core Metrics Comparison Across Datasets'
            )
            print(f"✅ Comparison plot saved to: {comparison_dir}")
        else:
            print(f"⚠️  No summary data found in any dataset")
        
        # # Create individual file plots for each dataset
        # print(f"📊 Creating individual file plots for each dataset")
        # self.plot_individual_files(results, self.output_dir)
        
        # print(f"\n🎉 All visualizations saved to: {self.output_dir}")
        # print("📁 Generated files:")
        # for file in sorted(self.output_dir.rglob("*.png")):
        #     print(f"   - {file.relative_to(self.output_dir)}")

    def plot_individual_files(self, results: Dict[str, Dict], output_dir: Path):
        """Create individual plots for each file within each dataset."""
        for dataset_name, dataset_data in results.items():
            if 'files' not in dataset_data:
                continue
                
            print(f"📊 Creating individual file plots for dataset: {dataset_name}")
            
            # Create directory for this dataset
            dataset_dir = output_dir / 'individual_files' / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            
            # Calculate chance performance and max values for this dataset
            chance_metrics = self.calculate_chance_performance({dataset_name: dataset_data})
            max_values = self.calculate_maximum_values({dataset_name: dataset_data})
            
            # Get all files for this dataset
            files = list(dataset_data['files'].keys())
            
            # Define metrics to plot
            metrics = [
                ('second_accuracy', 'Second-wise Accuracy', 0, 1),
                ('macro_f1', 'Macro F1 (Unweighted)', 0, 1),
                ('mAP', 'mean Average Precision (mAP)', 0, 1),
                ('mutual_info_gt_vs_pred', 'Mutual Info (GT vs Pred)', 0, 1),
                ('MCC', 'Matthew\'s Correlation Coefficient (MCC)', -1, 1),
            ]
            
            # Create figure for individual file plots
            fig, axes = plt.subplots(1, 5, figsize=(28, 12))
            axes = axes.flatten()
            
            for idx, (metric_key, title, y_min, y_max) in enumerate(metrics):
                ax = axes[idx]
                
                # Extract values for this metric from all files
                values = []
                for filename in files:
                    file_data = dataset_data['files'][filename]
                    value = file_data.get(metric_key, 0)
                    values.append(value)
                
                # Create bars for each file
                colors = sns.color_palette("husl", len(files))
                bars = ax.bar(range(len(files)), values, color=colors, alpha=0.8)
                
                ax.set_title(f'{title} - {dataset_name}', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric_key.replace('_', ' ').title())
                ax.set_xticks(range(len(files)))
                ax.set_xticklabels([f.split('_')[-1] if '_' in f else f for f in files], 
                                 rotation=45, ha='right')
                
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)
                elif y_min is not None:
                    ax.set_ylim(bottom=y_min)
                
                # Add chance performance and maximum value lines
                if 'boundary' not in metric_key.lower():
                    # Calculate chance performance for this specific file
                    file_chance_vals = []
                    file_max_vals = []
                    
                    for filename in files:
                        file_data = dataset_data['files'][filename]
                        temporal_coverage = file_data.get('temporal_coverage', {})
                        
                        if temporal_coverage:
                            total_duration = sum(coverage.get('gt_duration', 0) for coverage in temporal_coverage.values())
                            class_proportions = []
                            
                            for behavior, coverage in temporal_coverage.items():
                                if total_duration > 0:
                                    proportion = coverage.get('gt_duration', 0) / total_duration
                                    if proportion > 0:
                                        class_proportions.append(proportion)
                            
                            if class_proportions:
                                # Normalize proportions to sum to 1.0
                                total_proportion = sum(class_proportions)
                                if total_proportion > 0:
                                    class_proportions = [p / total_proportion for p in class_proportions]
                                
                                # Calculate chance performance for this file
                                if metric_key == 'second_accuracy':
                                    chance_val = sum(p * p for p in class_proportions)
                                elif metric_key == 'macro_f1':
                                    # For random classifier, macro F1 = average of class proportions
                                    # (precision = recall = class proportion for each class)
                                    chance_val = np.mean(class_proportions)
                                elif metric_key == 'weighted_f1':
                                    chance_val = sum(p * p for p in class_proportions)
                                elif metric_key == 'mAP':
                                    # For random classifier, mAP = average of class proportions
                                    chance_val = np.mean(class_proportions)
                                else:
                                    chance_val = 0
                                
                                # Calculate maximum mutual information for this file
                                if metric_key == 'mutual_info_gt_vs_pred':
                                    from collections import Counter
                                    file_labels = []
                                    for behavior, coverage in temporal_coverage.items():
                                        gt_duration = coverage.get('gt_duration', 0)
                                        if gt_duration > 0:
                                            file_labels.extend([behavior] * int(gt_duration))
                                    
                                    if file_labels:
                                        label_counts = Counter(file_labels)
                                        total_labels = len(file_labels)
                                        entropy = 0
                                        for count in label_counts.values():
                                            p = count / total_labels
                                            if p > 0:
                                                entropy -= p * np.log2(p)
                                        max_val = entropy
                                    else:
                                        max_val = 1.0
                                elif metric_key == 'MCC':
                                    max_val = 1.0
                                else:
                                    max_val = 0
                            else:
                                chance_val = 0
                                max_val = 1.0 if metric_key in ['mutual_info_gt_vs_pred', 'MCC'] else 0
                        else:
                            chance_val = 0
                            max_val = 1.0 if metric_key in ['mutual_info_gt_vs_pred', 'MCC'] else 0
                        
                        file_chance_vals.append(chance_val)
                        file_max_vals.append(max_val)
                    
                    # Add chance performance lines for each file
                    for i, (bar, chance_val) in enumerate(zip(bars, file_chance_vals)):
                        if chance_val > 0 and not metric_key.startswith('boundary') and metric_key != 'mAP':
                            ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], 
                                   [chance_val, chance_val], 
                                   color='red', linewidth=3, alpha=0.9,
                                   label='Chance Performance' if i == 0 and idx == 0 else None)
                        if metric_key == 'mutual_info_gt_vs_pred' or metric_key == 'mAP':
                            ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], 
                                   [0, 0], 
                                   color='red', linewidth=3, alpha=0.9,
                                   label='Chance Performance' if i == 0 and idx == 0 else None)
                    
                    # Add maximum value lines for each file
                    for i, (bar, max_val) in enumerate(zip(bars, file_max_vals)):
                        if metric_key == 'mutual_info_gt_vs_pred' and max_val > 0:
                            ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], 
                                   [max_val, max_val], 
                                   color='green', linewidth=2, alpha=0.8, linestyle='--',
                                   label='Max Possible' if i == 0 and idx == 3 else None)
                        elif metric_key == 'MCC':
                            ax.plot([bar.get_x(), bar.get_x() + bar.get_width()], 
                                   [1.0, 1.0], 
                                   color='green', linewidth=2, alpha=0.8, linestyle='--',
                                   label='Max Possible' if i == 0 and idx == 4 else None)
                
                # Add value labels on bars
                for bar, val in zip(bars, values):
                    label = f'{val:.3f}'
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(values) * 0.01 if values else 0.01), 
                           label, ha='center', va='bottom', fontweight='bold')
            
            # Hide any extra subplots
            for i in range(len(metrics), len(axes)):
                axes[i].set_visible(False)
            
            # Create legend - collect handles from all subplots
            file_legend_handles = [Patch(facecolor=colors[i], label=files[i]) for i in range(len(files))]
            
            # Collect all handles and labels from all subplots
            all_handles = []
            all_labels = []
            seen_labels = set()
            
            for ax in axes:
                if ax.get_visible():  # Only process visible subplots
                    handles, labels = ax.get_legend_handles_labels()
                    for handle, label in zip(handles, labels):
                        if label not in seen_labels:
                            all_handles.append(handle)
                            all_labels.append(label)
                            seen_labels.add(label)
            
            # Combine file handles with other handles
            all_handles = file_legend_handles + all_handles
            all_labels = files + all_labels
            
            if all_handles:
                fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.02), 
                          fontsize=12, ncol=min(len(all_handles), 8))
            
            # Add title
            fig.suptitle(f'Individual File Performance: {dataset_name}', fontsize=20, fontweight='bold')
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.08, 1, 0.97])
            
            # Save plot
            plt.savefig(dataset_dir / f'{dataset_name}_individual_files.png', dpi=300, bbox_inches='tight')
            plt.savefig(dataset_dir / f'{dataset_name}_individual_files.pdf', bbox_inches='tight')
            plt.close()
            
            print(f"✅ Individual file plots saved to: {dataset_dir}")

def main():
    parser = argparse.ArgumentParser(description='Generate comparative visualizations for action segmentation evaluation')
    parser.add_argument('--results-dir', required=True, 
                       help='Directory containing subdirectories with evaluation_results.json files')
    parser.add_argument('--output-dir', default='./comparative_plots',
                       help='Directory to save plots (default: ./comparative_plots)')
    args = parser.parse_args()
    visualizer = ComparativeVisualizer(args.output_dir)
    visualizer.generate_all_plots(Path(args.results_dir))

if __name__ == "__main__":
    main()