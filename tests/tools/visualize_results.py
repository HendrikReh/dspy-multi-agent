#!/usr/bin/env python
"""Visualize model test results with charts and graphs."""
import json
import argparse
from pathlib import Path
import sys
from typing import Dict, List, Any
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from utils.model_configs import MODEL_CONFIGS, ModelFamily, ModelCapability


class ResultsVisualizer:
    """Visualize test results from model comparisons."""
    
    def __init__(self, results_file: str):
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def extract_metrics_df(self) -> pd.DataFrame:
        """Extract metrics into a DataFrame."""
        records = []
        
        if "results" in self.data:  # Batch test format
            all_results = self.data["results"]["all_results"]
            for result in all_results:
                if result["status"] == "success" and result["result"]:
                    metrics = result["result"]["metrics"]
                    model_config = MODEL_CONFIGS.get(result["model"])
                    
                    records.append({
                        "model": result["model"],
                        "query_id": result["query_id"],
                        "family": model_config.family.value if model_config else "unknown",
                        "capability": model_config.capability.value if model_config else "unknown",
                        "time_to_first_token": metrics["time_to_first_token"],
                        "total_time": metrics["total_time"],
                        "tokens_per_second": metrics["tokens_per_second"],
                        "tokens_generated": metrics["tokens_generated"],
                    })
        
        elif "results" in self.data and isinstance(self.data["results"], list):  # Test series format
            for test_result in self.data["results"]:
                for model_name, model_result in test_result["model_results"].items():
                    if model_result and "metrics" in model_result:
                        metrics = model_result["metrics"]
                        model_config = MODEL_CONFIGS.get(model_name)
                        
                        records.append({
                            "model": model_name,
                            "query_id": test_result["query"]["id"],
                            "family": model_config.family.value if model_config else "unknown",
                            "capability": model_config.capability.value if model_config else "unknown",
                            "time_to_first_token": metrics["time_to_first_token"],
                            "total_time": metrics["total_time"],
                            "tokens_per_second": metrics["tokens_per_second"],
                            "tokens_generated": metrics["tokens_generated"],
                        })
        
        return pd.DataFrame(records)
    
    def plot_performance_comparison(self, df: pd.DataFrame):
        """Create performance comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16)
        
        # 1. Time to First Token by Model
        ax1 = axes[0, 0]
        model_order = ["o3", "o3-mini-low", "o3-mini-medium", "o3-mini-high",
                      "o4", "o4-mini-low", "o4-mini-medium", "o4-mini-high"]
        existing_models = [m for m in model_order if m in df['model'].unique()]
        
        ttft_data = df.groupby('model')['time_to_first_token'].mean().reindex(existing_models)
        ttft_data.plot(kind='bar', ax=ax1, color=['#FF6B6B' if 'o3' in m else '#4ECDC4' for m in existing_models])
        ax1.set_title("Average Time to First Token")
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Tokens per Second by Model
        ax2 = axes[0, 1]
        tps_data = df.groupby('model')['tokens_per_second'].mean().reindex(existing_models)
        tps_data.plot(kind='bar', ax=ax2, color=['#FF6B6B' if 'o3' in m else '#4ECDC4' for m in existing_models])
        ax2.set_title("Average Tokens per Second")
        ax2.set_xlabel("Model")
        ax2.set_ylabel("Tokens/sec")
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Performance by Capability Level
        ax3 = axes[1, 0]
        capability_order = ["low", "medium", "high", "full"]
        perf_by_cap = df.groupby('capability').agg({
            'time_to_first_token': 'mean',
            'tokens_per_second': 'mean'
        }).reindex(capability_order)
        
        x = np.arange(len(capability_order))
        width = 0.35
        
        ax3.bar(x - width/2, perf_by_cap['time_to_first_token'], width, label='Time to First Token')
        ax3_twin = ax3.twinx()
        ax3_twin.bar(x + width/2, perf_by_cap['tokens_per_second'], width, label='Tokens/sec', color='orange')
        
        ax3.set_xlabel('Capability Level')
        ax3.set_ylabel('Time to First Token (s)', color='blue')
        ax3_twin.set_ylabel('Tokens per Second', color='orange')
        ax3.set_xticks(x)
        ax3.set_xticklabels(capability_order)
        ax3.set_title("Performance by Capability Level")
        
        # 4. O3 vs O4 Family Comparison
        ax4 = axes[1, 1]
        family_comparison = df.groupby('family').agg({
            'time_to_first_token': ['mean', 'std'],
            'total_time': ['mean', 'std'],
            'tokens_per_second': ['mean', 'std']
        })
        
        metrics = ['Time to First Token', 'Total Time', 'Tokens/Second']
        o3_means = [
            family_comparison.loc['o3', ('time_to_first_token', 'mean')] if 'o3' in family_comparison.index else 0,
            family_comparison.loc['o3', ('total_time', 'mean')] if 'o3' in family_comparison.index else 0,
            family_comparison.loc['o3', ('tokens_per_second', 'mean')] if 'o3' in family_comparison.index else 0,
        ]
        o4_means = [
            family_comparison.loc['o4', ('time_to_first_token', 'mean')] if 'o4' in family_comparison.index else 0,
            family_comparison.loc['o4', ('total_time', 'mean')] if 'o4' in family_comparison.index else 0,
            family_comparison.loc['o4', ('tokens_per_second', 'mean')] if 'o4' in family_comparison.index else 0,
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax4.bar(x - width/2, o3_means, width, label='O3 Family', color='#FF6B6B')
        ax4.bar(x + width/2, o4_means, width, label='O4 Family', color='#4ECDC4')
        
        ax4.set_xlabel('Metric')
        ax4.set_ylabel('Value')
        ax4.set_title('O3 vs O4 Family Comparison')
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics)
        ax4.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_heatmap(self, df: pd.DataFrame):
        """Create a heatmap of model performance across queries."""
        # Pivot data for heatmap
        pivot_ttft = df.pivot_table(
            values='time_to_first_token',
            index='model',
            columns='query_id',
            aggfunc='mean'
        )
        
        # Sort models by family and capability
        model_order = ["o3", "o3-mini-low", "o3-mini-medium", "o3-mini-high",
                      "o4", "o4-mini-low", "o4-mini-medium", "o4-mini-high"]
        existing_models = [m for m in model_order if m in pivot_ttft.index]
        pivot_ttft = pivot_ttft.reindex(existing_models)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        sns.heatmap(
            pivot_ttft,
            annot=True,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Time to First Token (s)'},
            ax=ax
        )
        
        ax.set_title('Time to First Token Heatmap: Models vs Queries', fontsize=14)
        ax.set_xlabel('Query ID')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        return fig
    
    def plot_speedup_matrix(self, df: pd.DataFrame):
        """Create speedup comparison matrix."""
        models = df['model'].unique()
        n_models = len(models)
        
        # Calculate average times for each model
        avg_times = df.groupby('model')['total_time'].mean()
        
        # Create speedup matrix
        speedup_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if model1 in avg_times and model2 in avg_times:
                    speedup_matrix[i, j] = avg_times[model1] / avg_times[model2]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(speedup_matrix, cmap='RdBu_r', center=1.0, vmin=0.5, vmax=2.0)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Speedup Factor', rotation=270, labelpad=20)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(n_models))
        ax.set_yticks(np.arange(n_models))
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.set_yticklabels(models)
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                text = ax.text(j, i, f'{speedup_matrix[i, j]:.2f}',
                             ha='center', va='center', color='black' if abs(speedup_matrix[i, j] - 1) < 0.5 else 'white')
        
        ax.set_title('Model Speedup Comparison Matrix\n(Row model time / Column model time)', fontsize=14)
        ax.set_xlabel('Compared to Model')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_dir: str = "visualization_results"):
        """Generate complete visualization report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Extract data
        df = self.extract_metrics_df()
        
        if df.empty:
            print("No data to visualize!")
            return
        
        print(f"Loaded {len(df)} test results")
        
        # Generate plots
        print("Generating performance comparison...")
        fig1 = self.plot_performance_comparison(df)
        fig1.savefig(output_path / "performance_comparison.png", dpi=300, bbox_inches='tight')
        
        print("Generating heatmap...")
        fig2 = self.plot_heatmap(df)
        fig2.savefig(output_path / "performance_heatmap.png", dpi=300, bbox_inches='tight')
        
        print("Generating speedup matrix...")
        fig3 = self.plot_speedup_matrix(df)
        fig3.savefig(output_path / "speedup_matrix.png", dpi=300, bbox_inches='tight')
        
        # Generate statistics summary
        print("Generating statistics...")
        stats = self.generate_statistics(df)
        
        with open(output_path / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate markdown report
        self.generate_markdown_report(df, stats, output_path)
        
        print(f"\nVisualization complete! Results saved to {output_path}/")
    
    def generate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate statistical summary."""
        stats = {
            "overall": {
                "total_tests": len(df),
                "unique_models": df['model'].nunique(),
                "unique_queries": df['query_id'].nunique(),
            },
            "by_model": {},
            "by_family": {},
            "by_capability": {},
        }
        
        # Statistics by model
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            stats["by_model"][model] = {
                "count": len(model_data),
                "avg_time_to_first_token": model_data['time_to_first_token'].mean(),
                "std_time_to_first_token": model_data['time_to_first_token'].std(),
                "avg_total_time": model_data['total_time'].mean(),
                "avg_tokens_per_second": model_data['tokens_per_second'].mean(),
            }
        
        # Statistics by family
        for family in df['family'].unique():
            family_data = df[df['family'] == family]
            stats["by_family"][family] = {
                "count": len(family_data),
                "models": family_data['model'].unique().tolist(),
                "avg_time_to_first_token": family_data['time_to_first_token'].mean(),
                "avg_tokens_per_second": family_data['tokens_per_second'].mean(),
            }
        
        # Statistics by capability
        for capability in df['capability'].unique():
            cap_data = df[df['capability'] == capability]
            stats["by_capability"][capability] = {
                "count": len(cap_data),
                "avg_time_to_first_token": cap_data['time_to_first_token'].mean(),
                "avg_tokens_per_second": cap_data['tokens_per_second'].mean(),
            }
        
        return stats
    
    def generate_markdown_report(self, df: pd.DataFrame, stats: Dict[str, Any], output_path: Path):
        """Generate markdown report."""
        report = ["# Model Test Results Report\n"]
        report.append(f"Generated: {self.data.get('timestamp', 'Unknown')}\n")
        
        # Overall statistics
        report.append("## Overall Statistics\n")
        report.append(f"- Total tests: {stats['overall']['total_tests']}")
        report.append(f"- Models tested: {stats['overall']['unique_models']}")
        report.append(f"- Queries tested: {stats['overall']['unique_queries']}\n")
        
        # Performance rankings
        report.append("## Performance Rankings\n")
        
        # Fastest time to first token
        report.append("### Fastest Time to First Token")
        ttft_ranking = df.groupby('model')['time_to_first_token'].mean().sort_values()
        for i, (model, time) in enumerate(ttft_ranking.head(5).items(), 1):
            report.append(f"{i}. **{model}**: {time:.3f}s")
        report.append("")
        
        # Highest tokens per second
        report.append("### Highest Tokens per Second")
        tps_ranking = df.groupby('model')['tokens_per_second'].mean().sort_values(ascending=False)
        for i, (model, tps) in enumerate(tps_ranking.head(5).items(), 1):
            report.append(f"{i}. **{model}**: {tps:.1f} tokens/sec")
        report.append("")
        
        # Family comparison
        report.append("## Family Comparison\n")
        for family, family_stats in stats['by_family'].items():
            report.append(f"### {family.upper()} Family")
            report.append(f"- Models: {', '.join(family_stats['models'])}")
            report.append(f"- Avg Time to First Token: {family_stats['avg_time_to_first_token']:.3f}s")
            report.append(f"- Avg Tokens/Second: {family_stats['avg_tokens_per_second']:.1f}\n")
        
        # Visualizations
        report.append("## Visualizations\n")
        report.append("![Performance Comparison](performance_comparison.png)\n")
        report.append("![Performance Heatmap](performance_heatmap.png)\n")
        report.append("![Speedup Matrix](speedup_matrix.png)\n")
        
        # Write report
        with open(output_path / "report.md", 'w') as f:
            f.write('\n'.join(report))


def main():
    """Run visualization."""
    parser = argparse.ArgumentParser(description="Visualize model test results")
    parser.add_argument(
        "results_file",
        help="Path to results JSON file"
    )
    parser.add_argument(
        "--output",
        default="visualization_results",
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file '{args.results_file}' not found!")
        sys.exit(1)
    
    # Create visualizer
    visualizer = ResultsVisualizer(args.results_file)
    
    # Generate report
    visualizer.generate_report(args.output)


if __name__ == "__main__":
    # Check dependencies
    try:
        import matplotlib
        import seaborn
    except ImportError:
        print("Installing visualization dependencies...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib", "seaborn"])
    
    main()