#!/usr/bin/env python
"""Visualize model comparison results from test_model_comparison.py output."""
import json
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class ComparisonVisualizer:
    """Visualize model comparison test results."""
    
    def __init__(self, results_file: str):
        with open(results_file, 'r') as f:
            self.data = json.load(f)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def extract_metrics_df(self) -> pd.DataFrame:
        """Extract performance metrics from comparison data."""
        records = []
        
        if "performance_comparisons" in self.data:
            # Extract unique model performances
            model_metrics = {}
            
            for comparison in self.data["performance_comparisons"]:
                # Process model_a
                model_a = comparison["model_a"]
                if model_a["name"] not in model_metrics:
                    model_metrics[model_a["name"]] = {
                        "time_to_first_token": [],
                        "total_time": [],
                        "tokens_generated": [],
                        "tokens_per_second": []
                    }
                model_metrics[model_a["name"]]["time_to_first_token"].append(model_a["time_to_first_token"])
                model_metrics[model_a["name"]]["total_time"].append(model_a["total_time"])
                model_metrics[model_a["name"]]["tokens_generated"].append(model_a["tokens_generated"])
                model_metrics[model_a["name"]]["tokens_per_second"].append(
                    model_a["tokens_generated"] / model_a["total_time"] if model_a["total_time"] > 0 else 0
                )
                
                # Process model_b
                model_b = comparison["model_b"]
                if model_b["name"] not in model_metrics:
                    model_metrics[model_b["name"]] = {
                        "time_to_first_token": [],
                        "total_time": [],
                        "tokens_generated": [],
                        "tokens_per_second": []
                    }
                model_metrics[model_b["name"]]["time_to_first_token"].append(model_b["time_to_first_token"])
                model_metrics[model_b["name"]]["total_time"].append(model_b["total_time"])
                model_metrics[model_b["name"]]["tokens_generated"].append(model_b["tokens_generated"])
                model_metrics[model_b["name"]]["tokens_per_second"].append(
                    model_b["tokens_generated"] / model_b["total_time"] if model_b["total_time"] > 0 else 0
                )
            
            # Create records for DataFrame
            for model_name, metrics in model_metrics.items():
                records.append({
                    "model": model_name,
                    "time_to_first_token": np.mean(metrics["time_to_first_token"]),
                    "total_time": np.mean(metrics["total_time"]),
                    "tokens_generated": np.mean(metrics["tokens_generated"]),
                    "tokens_per_second": np.mean(metrics["tokens_per_second"]),
                })
        
        return pd.DataFrame(records)
    
    def plot_performance_comparison(self, df: pd.DataFrame):
        """Create performance comparison plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Model Performance Comparison", fontsize=16)
        
        # Sort models for consistent ordering
        model_order = ["gpt-4o", "gpt-4o-mini", "o3", "o3-mini", "o4-mini"]
        df = df.set_index('model').reindex(model_order).reset_index()
        
        # 1. Time to First Token by Model
        ax1 = axes[0, 0]
        colors = ['#FF6B6B' if 'o3' in m or 'o4' in m else '#4ECDC4' for m in df['model']]
        df.plot(x='model', y='time_to_first_token', kind='bar', ax=ax1, color=colors, legend=False)
        ax1.set_title("Average Time to First Token")
        ax1.set_xlabel("Model")
        ax1.set_ylabel("Time (seconds)")
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Tokens per Second by Model
        ax2 = axes[0, 1]
        df.plot(x='model', y='tokens_per_second', kind='bar', ax=ax2, color=colors, legend=False)
        ax2.set_title("Average Tokens per Second")
        ax2.set_xlabel("Model")
        ax2.set_ylabel("Tokens/sec")
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Total Time by Model
        ax3 = axes[1, 0]
        df.plot(x='model', y='total_time', kind='bar', ax=ax3, color=colors, legend=False)
        ax3.set_title("Average Total Time")
        ax3.set_xlabel("Model")
        ax3.set_ylabel("Time (seconds)")
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Model Performance Summary
        ax4 = axes[1, 1]
        # Create a scatter plot of time vs throughput
        for i, row in df.iterrows():
            color = '#FF6B6B' if 'o3' in row['model'] or 'o4' in row['model'] else '#4ECDC4'
            ax4.scatter(row['total_time'], row['tokens_per_second'], 
                       s=200, color=color, alpha=0.7, edgecolors='black', linewidth=2)
            ax4.annotate(row['model'], (row['total_time'], row['tokens_per_second']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        ax4.set_xlabel('Total Time (seconds)')
        ax4.set_ylabel('Tokens per Second')
        ax4.set_title('Performance Trade-off: Speed vs Throughput')
        ax4.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#4ECDC4', label='GPT Models'),
            Patch(facecolor='#FF6B6B', label='O3/o4 Models')
        ]
        ax4.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def plot_speedup_matrix(self, df: pd.DataFrame):
        """Create speedup comparison matrix from performance comparisons."""
        if "performance_comparisons" not in self.data:
            return None
            
        models = df['model'].tolist()
        n_models = len(models)
        
        # Create speedup matrix
        speedup_matrix = np.ones((n_models, n_models))
        
        # Fill matrix from comparisons
        for comp in self.data["performance_comparisons"]:
            model_a = comp["model_a"]["name"]
            model_b = comp["model_b"]["name"]
            speedup = comp["comparison"]["speedup_total"]
            
            if model_a in models and model_b in models:
                i = models.index(model_a)
                j = models.index(model_b)
                speedup_matrix[i, j] = speedup
                speedup_matrix[j, i] = 1.0 / speedup if speedup > 0 else 0
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(speedup_matrix, cmap='RdBu_r', vmin=0, vmax=2.0)
        
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
                             ha='center', va='center', 
                             color='white' if abs(speedup_matrix[i, j] - 1) > 0.5 else 'black')
        
        ax.set_title('Model Speedup Comparison Matrix\n(Row model time / Column model time)', fontsize=14)
        ax.set_xlabel('Compared to Model')
        ax.set_ylabel('Model')
        
        plt.tight_layout()
        return fig
    
    def generate_report(self, output_dir: str = "comparison_visualization"):
        """Generate complete visualization report."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Extract data
        df = self.extract_metrics_df()
        
        if df.empty:
            print("No performance data found in the results file!")
            return
        
        print(f"Loaded performance data for {len(df)} models")
        print("\nModel Performance Summary:")
        print(df.to_string(index=False))
        
        # Generate plots
        print("\nGenerating visualizations...")
        
        # Performance comparison
        fig1 = self.plot_performance_comparison(df)
        fig1.savefig(output_path / "performance_comparison.png", dpi=300, bbox_inches='tight')
        print("✓ Created performance_comparison.png")
        
        # Speedup matrix
        fig2 = self.plot_speedup_matrix(df)
        if fig2:
            fig2.savefig(output_path / "speedup_matrix.png", dpi=300, bbox_inches='tight')
            print("✓ Created speedup_matrix.png")
        
        # Generate markdown report
        self.generate_markdown_report(df, output_path)
        print("✓ Created report.md")
        
        print(f"\nVisualization complete! Results saved to {output_path}/")
    
    def generate_markdown_report(self, df: pd.DataFrame, output_path: Path):
        """Generate markdown report."""
        report = ["# Model Comparison Results Report\n"]
        report.append(f"Generated from: {self.data.get('timestamp', 'Unknown')}\n")
        
        # Models tested
        report.append("## Models Tested\n")
        for model in self.data.get('models_tested', []):
            report.append(f"- {model}")
        report.append("")
        
        # Performance summary
        report.append("## Performance Summary\n")
        report.append("| Model | Time to First Token (s) | Total Time (s) | Tokens/Second | Tokens Generated |")
        report.append("|-------|-------------------------|----------------|---------------|------------------|")
        
        for _, row in df.iterrows():
            report.append(f"| {row['model']} | {row['time_to_first_token']:.3f} | "
                         f"{row['total_time']:.2f} | {row['tokens_per_second']:.1f} | "
                         f"{row['tokens_generated']:.0f} |")
        report.append("")
        
        # Performance rankings
        report.append("## Performance Rankings\n")
        
        # Fastest time to first token
        report.append("### Fastest Time to First Token")
        df_sorted = df.sort_values('time_to_first_token')
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            report.append(f"{i}. **{row['model']}**: {row['time_to_first_token']:.3f}s")
        report.append("")
        
        # Highest tokens per second
        report.append("### Highest Tokens per Second")
        df_sorted = df.sort_values('tokens_per_second', ascending=False)
        for i, (_, row) in enumerate(df_sorted.head(5).iterrows(), 1):
            report.append(f"{i}. **{row['model']}**: {row['tokens_per_second']:.1f} tokens/sec")
        report.append("")
        
        # Key insights
        report.append("## Key Insights\n")
        fastest_ttft = df.loc[df['time_to_first_token'].idxmin()]
        fastest_overall = df.loc[df['total_time'].idxmin()]
        highest_throughput = df.loc[df['tokens_per_second'].idxmax()]
        
        report.append(f"- **Fastest Time to First Token**: {fastest_ttft['model']} ({fastest_ttft['time_to_first_token']:.3f}s)")
        report.append(f"- **Fastest Overall**: {fastest_overall['model']} ({fastest_overall['total_time']:.2f}s)")
        report.append(f"- **Highest Throughput**: {highest_throughput['model']} ({highest_throughput['tokens_per_second']:.1f} tokens/sec)")
        report.append("")
        
        # Visualizations
        report.append("## Visualizations\n")
        report.append("![Performance Comparison](performance_comparison.png)\n")
        report.append("![Speedup Matrix](speedup_matrix.png)\n")
        
        # Write report
        with open(output_path / "report.md", 'w') as f:
            f.write('\n'.join(report))

def main():
    """Run visualization."""
    parser = argparse.ArgumentParser(description="Visualize model comparison results")
    parser.add_argument(
        "results_file",
        help="Path to model comparison results JSON file"
    )
    parser.add_argument(
        "--output",
        default="comparison_visualization",
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.results_file).exists():
        print(f"Error: Results file '{args.results_file}' not found!")
        return
    
    # Create visualizer
    visualizer = ComparisonVisualizer(args.results_file)
    
    # Generate report
    visualizer.generate_report(args.output)

if __name__ == "__main__":
    main()