"""Comprehensive report generator for model comparison tests."""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from jinja2 import Template
import base64
from io import BytesIO

class ComprehensiveReportGenerator:
    """Generate comprehensive reports from model comparison tests."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_dir = self.output_dir / f"model_comparison_{self.timestamp}"
        self.report_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.llm_outputs_dir = self.report_dir / "llm_outputs"
        self.visualizations_dir = self.report_dir / "visualizations"
        self.data_dir = self.report_dir / "data"
        
        for dir in [self.llm_outputs_dir, self.visualizations_dir, self.data_dir]:
            dir.mkdir(exist_ok=True)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Store test data
        self.test_data = {
            "timestamp": self.timestamp,
            "models": [],
            "queries": [],
            "results": [],
            "comparisons": [],
            "llm_outputs": {}
        }
    
    def add_test_result(self, model: str, query: Dict[str, Any], result: Dict[str, Any], 
                       llm_output: Optional[str] = None):
        """Add a test result to the report."""
        # Store model and query
        if model not in self.test_data["models"]:
            self.test_data["models"].append(model)
        
        if query not in self.test_data["queries"]:
            self.test_data["queries"].append(query)
        
        # Store result
        test_result = {
            "model": model,
            "query": query,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.test_data["results"].append(test_result)
        
        # Store LLM output if provided
        if llm_output:
            output_filename = f"{model}_{len(self.test_data['results'])}.txt"
            output_path = self.llm_outputs_dir / output_filename
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"Model: {model}\n")
                f.write(f"Query: {query.get('request', 'N/A')}\n")
                f.write(f"Timestamp: {test_result['timestamp']}\n")
                f.write("="*80 + "\n\n")
                # Write the actual LLM output (no placeholder text)
                f.write(llm_output)
            
            # Store reference
            if model not in self.test_data["llm_outputs"]:
                self.test_data["llm_outputs"][model] = []
            self.test_data["llm_outputs"][model].append(str(output_path))
    
    def add_comparison(self, model_a: str, model_b: str, comparison: Dict[str, Any]):
        """Add a model comparison result."""
        self.test_data["comparisons"].append({
            "model_a": model_a,
            "model_b": model_b,
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        })
    
    def generate_performance_visualizations(self) -> Dict[str, str]:
        """Generate performance comparison visualizations."""
        # Extract metrics data
        metrics_data = []
        for result in self.test_data["results"]:
            if "metrics" in result["result"]:
                metrics = result["result"]["metrics"]
                metrics_data.append({
                    "model": result["model"],
                    "time_to_first_token": metrics.get("time_to_first_token", 0),
                    "total_time": metrics.get("total_time", 0),
                    "tokens_generated": metrics.get("tokens_generated", 0),
                    "tokens_per_second": metrics.get("tokens_per_second", 0)
                })
        
        if not metrics_data:
            return {}
        
        df = pd.DataFrame(metrics_data)
        
        # Group by model and calculate averages
        df_avg = df.groupby('model').mean().reset_index()
        
        visualizations = {}
        
        # 1. Performance Overview
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # Time to First Token
        ax1 = axes[0, 0]
        colors = ['#FF6B6B' if 'o3' in m or 'o4' in m else '#4ECDC4' for m in df_avg['model']]
        df_avg.plot(x='model', y='time_to_first_token', kind='bar', ax=ax1, color=colors, legend=False)
        ax1.set_title('Average Time to First Token')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Tokens per Second
        ax2 = axes[0, 1]
        df_avg.plot(x='model', y='tokens_per_second', kind='bar', ax=ax2, color=colors, legend=False)
        ax2.set_title('Average Tokens per Second')
        ax2.set_ylabel('Tokens/sec')
        ax2.tick_params(axis='x', rotation=45)
        
        # Total Time
        ax3 = axes[1, 0]
        df_avg.plot(x='model', y='total_time', kind='bar', ax=ax3, color=colors, legend=False)
        ax3.set_title('Average Total Time')
        ax3.set_ylabel('Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Performance Scatter
        ax4 = axes[1, 1]
        for i, row in df_avg.iterrows():
            color = '#FF6B6B' if 'o3' in row['model'] or 'o4' in row['model'] else '#4ECDC4'
            ax4.scatter(row['total_time'], row['tokens_per_second'], 
                       s=200, color=color, alpha=0.7, edgecolors='black', linewidth=2)
            ax4.annotate(row['model'], (row['total_time'], row['tokens_per_second']), 
                        xytext=(5, 5), textcoords='offset points')
        ax4.set_xlabel('Total Time (seconds)')
        ax4.set_ylabel('Tokens per Second')
        ax4.set_title('Performance Trade-off: Speed vs Throughput')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        perf_path = self.visualizations_dir / "performance_overview.png"
        fig.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations['performance_overview'] = str(perf_path)
        
        # 2. Speedup Matrix
        if self.test_data["comparisons"]:
            models = df_avg['model'].tolist()
            n_models = len(models)
            speedup_matrix = np.ones((n_models, n_models))
            
            for comp in self.test_data["comparisons"]:
                if "comparison" in comp and "speedup_total" in comp["comparison"].get("comparison", {}):
                    model_a = comp["model_a"]
                    model_b = comp["model_b"]
                    speedup = comp["comparison"]["comparison"]["speedup_total"]
                    
                    if model_a in models and model_b in models:
                        i = models.index(model_a)
                        j = models.index(model_b)
                        speedup_matrix[i, j] = speedup
                        speedup_matrix[j, i] = 1.0 / speedup if speedup > 0 else 0
            
            fig, ax = plt.subplots(figsize=(10, 8))
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
            
            ax.set_title('Model Speedup Comparison Matrix\n(Row model time / Column model time)')
            plt.tight_layout()
            
            speedup_path = self.visualizations_dir / "speedup_matrix.png"
            fig.savefig(speedup_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['speedup_matrix'] = str(speedup_path)
        
        # 3. Token Generation Distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        df.boxplot(column='tokens_generated', by='model', ax=ax)
        ax.set_title('Token Generation Distribution by Model')
        ax.set_xlabel('Model')
        ax.set_ylabel('Tokens Generated')
        plt.suptitle('')  # Remove default title
        
        tokens_path = self.visualizations_dir / "token_distribution.png"
        fig.savefig(tokens_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualizations['token_distribution'] = str(tokens_path)
        
        return visualizations
    
    def generate_similarity_visualizations(self) -> Dict[str, str]:
        """Generate content similarity visualizations."""
        visualizations = {}
        
        if not self.test_data["comparisons"]:
            return visualizations
        
        # Extract similarity data
        similarity_data = []
        for comp in self.test_data["comparisons"]:
            if "similarity" in comp["comparison"]:
                sim = comp["comparison"]["similarity"]
                similarity_data.append({
                    "comparison": f"{comp['model_a']} vs {comp['model_b']}",
                    "overall_similarity": sim.get("overall_similarity", 0) * 100,
                    "key_points_overlap": sim.get("key_points_overlap", 0) * 100,
                    "vocabulary_overlap": sim.get("vocabulary_overlap", 0) * 100,
                    "source_overlap": sim.get("source_overlap", 0) * 100
                })
        
        if similarity_data:
            df_sim = pd.DataFrame(similarity_data)
            
            # Similarity heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Prepare data for heatmap
            metrics = ['overall_similarity', 'key_points_overlap', 'vocabulary_overlap', 'source_overlap']
            heatmap_data = df_sim[metrics].T
            heatmap_data.columns = df_sim['comparison']
            
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Similarity %'}, ax=ax)
            ax.set_title('Content Similarity Analysis')
            ax.set_xlabel('Model Comparison')
            ax.set_ylabel('Similarity Metric')
            
            plt.tight_layout()
            sim_path = self.visualizations_dir / "similarity_heatmap.png"
            fig.savefig(sim_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualizations['similarity_heatmap'] = str(sim_path)
        
        return visualizations
    
    def save_data_files(self):
        """Save all test data to JSON files."""
        # Save complete test data
        with open(self.data_dir / "complete_test_data.json", 'w') as f:
            json.dump(self.test_data, f, indent=2, default=str)
        
        # Save metrics summary
        metrics_summary = []
        for result in self.test_data["results"]:
            if "metrics" in result["result"]:
                metrics_summary.append({
                    "model": result["model"],
                    "query": result["query"]["request"][:50] + "...",
                    "metrics": result["result"]["metrics"]
                })
        
        with open(self.data_dir / "metrics_summary.json", 'w') as f:
            json.dump(metrics_summary, f, indent=2)
        
        # Save comparison summary
        comparison_summary = []
        for comp in self.test_data["comparisons"]:
            comparison_summary.append({
                "models": f"{comp['model_a']} vs {comp['model_b']}",
                "similarity": comp["comparison"].get("similarity", {}),
                "performance": comp["comparison"].get("comparison", {})
            })
        
        with open(self.data_dir / "comparison_summary.json", 'w') as f:
            json.dump(comparison_summary, f, indent=2)
    
    def image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 for embedding in HTML."""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def generate_html_report(self):
        """Generate comprehensive HTML report."""
        # Generate visualizations
        perf_viz = self.generate_performance_visualizations()
        sim_viz = self.generate_similarity_visualizations()
        
        # Save data files
        self.save_data_files()
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats()
        
        # HTML template
        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Model Comparison Report - {{ timestamp }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #333;
        }
        h1 {
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .summary-card {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #495057;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #007bff;
        }
        .model-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0;
        }
        .model-badge {
            background-color: #e9ecef;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
        }
        .model-badge.o3-model {
            background-color: #ff6b6b;
            color: white;
        }
        .model-badge.gpt-model {
            background-color: #4ecdc4;
            color: white;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #f8f9fa;
            font-weight: bold;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .visualization {
            margin: 20px 0;
            text-align: center;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .llm-output {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            padding: 15px;
            margin: 10px 0;
            white-space: pre-wrap;
            font-family: monospace;
            font-size: 12px;
            max-height: 400px;
            overflow-y: auto;
        }
        .comparison-section {
            background-color: #e7f5ff;
            border: 1px solid #a5d8ff;
            border-radius: 5px;
            padding: 15px;
            margin: 20px 0;
        }
        .footer {
            margin-top: 50px;
            text-align: center;
            color: #6c757d;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Comparison Report</h1>
        <p><strong>Generated:</strong> {{ timestamp }}</p>
        
        <h2>Executive Summary</h2>
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Models Tested</h3>
                <div class="value">{{ summary.models_count }}</div>
            </div>
            <div class="summary-card">
                <h3>Test Queries</h3>
                <div class="value">{{ summary.queries_count }}</div>
            </div>
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">{{ summary.total_tests }}</div>
            </div>
            <div class="summary-card">
                <h3>Fastest Model</h3>
                <div class="value">{{ summary.fastest_model }}</div>
            </div>
        </div>
        
        <h2>Models Tested</h2>
        <div class="model-list">
            {% for model in models %}
                <span class="model-badge {% if 'o3' in model or 'o4' in model %}o3-model{% else %}gpt-model{% endif %}">
                    {{ model }}
                </span>
            {% endfor %}
        </div>
        
        <h2>Performance Summary</h2>
        <table>
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Avg Time to First Token (s)</th>
                    <th>Avg Total Time (s)</th>
                    <th>Avg Tokens/Second</th>
                    <th>Avg Tokens Generated</th>
                </tr>
            </thead>
            <tbody>
                {% for stat in summary.model_stats %}
                <tr>
                    <td><strong>{{ stat.model }}</strong></td>
                    <td>{{ "%.3f"|format(stat.avg_time_to_first_token) }}</td>
                    <td>{{ "%.2f"|format(stat.avg_total_time) }}</td>
                    <td>{{ "%.1f"|format(stat.avg_tokens_per_second) }}</td>
                    <td>{{ "%.0f"|format(stat.avg_tokens_generated) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        <h2>Performance Visualizations</h2>
        {% if visualizations.performance_overview %}
        <div class="visualization">
            <h3>Performance Overview</h3>
            <img src="data:image/png;base64,{{ visualizations.performance_overview }}" alt="Performance Overview">
        </div>
        {% endif %}
        
        {% if visualizations.speedup_matrix %}
        <div class="visualization">
            <h3>Speedup Matrix</h3>
            <img src="data:image/png;base64,{{ visualizations.speedup_matrix }}" alt="Speedup Matrix">
        </div>
        {% endif %}
        
        {% if visualizations.token_distribution %}
        <div class="visualization">
            <h3>Token Generation Distribution</h3>
            <img src="data:image/png;base64,{{ visualizations.token_distribution }}" alt="Token Distribution">
        </div>
        {% endif %}
        
        <h2>Content Similarity Analysis</h2>
        {% if visualizations.similarity_heatmap %}
        <div class="visualization">
            <img src="data:image/png;base64,{{ visualizations.similarity_heatmap }}" alt="Similarity Heatmap">
        </div>
        {% endif %}
        
        <h2>Model Comparisons</h2>
        {% for comp in comparisons %}
        <div class="comparison-section">
            <h3>{{ comp.model_a }} vs {{ comp.model_b }}</h3>
            {% if comp.similarity %}
            <p><strong>Content Similarity:</strong></p>
            <ul>
                <li>Overall Similarity: {{ "%.1f"|format(comp.similarity.overall_similarity * 100) }}%</li>
                <li>Key Points Overlap: {{ "%.1f"|format(comp.similarity.key_points_overlap * 100) }}%</li>
                <li>Vocabulary Overlap: {{ "%.1f"|format(comp.similarity.vocabulary_overlap * 100) }}%</li>
            </ul>
            {% endif %}
            {% if comp.performance %}
            <p><strong>Performance Comparison:</strong></p>
            <ul>
                <li>First Token Speedup: {{ "%.2f"|format(comp.performance.speedup_first_token) }}x</li>
                <li>Total Time Speedup: {{ "%.2f"|format(comp.performance.speedup_total) }}x</li>
            </ul>
            {% endif %}
        </div>
        {% endfor %}
        
        <h2>Test Queries</h2>
        <ol>
            {% for query in queries %}
            <li>
                <strong>Request:</strong> {{ query.request }}<br>
                <strong>Target Audience:</strong> {{ query.target_audience }}<br>
                <strong>Max Sources:</strong> {{ query.max_sources }}
            </li>
            {% endfor %}
        </ol>
        
        <h2>LLM Output Samples</h2>
        <p>Full outputs are saved in the <code>llm_outputs</code> directory.</p>
        {% for model, outputs in llm_outputs.items() %}
        <h3>{{ model }}</h3>
        <p>Generated {{ outputs|length }} output(s). Files:</p>
        <ul>
            {% for output in outputs %}
            <li><code>{{ output }}</code></li>
            {% endfor %}
        </ul>
        {% endfor %}
        
        <h2>Data Files</h2>
        <ul>
            <li><code>data/complete_test_data.json</code> - Complete test data</li>
            <li><code>data/metrics_summary.json</code> - Performance metrics summary</li>
            <li><code>data/comparison_summary.json</code> - Model comparison summary</li>
        </ul>
        
        <div class="footer">
            <p>Report generated by DSPy Multi-Agent Model Comparison Tool</p>
            <p>{{ timestamp }}</p>
        </div>
    </div>
</body>
</html>
        """)
        
        # Prepare template data
        template_data = {
            "timestamp": self.timestamp,
            "models": self.test_data["models"],
            "queries": self.test_data["queries"],
            "comparisons": [
                {
                    "model_a": comp["model_a"],
                    "model_b": comp["model_b"],
                    "similarity": comp["comparison"].get("similarity", {}),
                    "performance": comp["comparison"].get("comparison", {})
                }
                for comp in self.test_data["comparisons"]
            ],
            "summary": summary_stats,
            "visualizations": {
                name: self.image_to_base64(Path(path))
                for name, path in {**perf_viz, **sim_viz}.items()
            },
            "llm_outputs": self.test_data["llm_outputs"]
        }
        
        # Generate HTML
        html_content = html_template.render(**template_data)
        
        # Save HTML report
        report_path = self.report_dir / f"report_{self.timestamp}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"\nComprehensive report generated:")
        print(f"  Report: {report_path}")
        print(f"  LLM Outputs: {self.llm_outputs_dir}")
        print(f"  Visualizations: {self.visualizations_dir}")
        print(f"  Data Files: {self.data_dir}")
        
        return str(report_path)
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """Calculate summary statistics."""
        # Extract metrics
        model_metrics = {}
        for result in self.test_data["results"]:
            model = result["model"]
            if model not in model_metrics:
                model_metrics[model] = {
                    "time_to_first_token": [],
                    "total_time": [],
                    "tokens_per_second": [],
                    "tokens_generated": []
                }
            
            if "metrics" in result["result"]:
                metrics = result["result"]["metrics"]
                model_metrics[model]["time_to_first_token"].append(metrics.get("time_to_first_token", 0))
                model_metrics[model]["total_time"].append(metrics.get("total_time", 0))
                model_metrics[model]["tokens_per_second"].append(metrics.get("tokens_per_second", 0))
                model_metrics[model]["tokens_generated"].append(metrics.get("tokens_generated", 0))
        
        # Calculate averages
        model_stats = []
        fastest_model = None
        fastest_time = float('inf')
        
        for model, metrics in model_metrics.items():
            avg_stats = {
                "model": model,
                "avg_time_to_first_token": np.mean(metrics["time_to_first_token"]) if metrics["time_to_first_token"] else 0,
                "avg_total_time": np.mean(metrics["total_time"]) if metrics["total_time"] else 0,
                "avg_tokens_per_second": np.mean(metrics["tokens_per_second"]) if metrics["tokens_per_second"] else 0,
                "avg_tokens_generated": np.mean(metrics["tokens_generated"]) if metrics["tokens_generated"] else 0
            }
            model_stats.append(avg_stats)
            
            if avg_stats["avg_total_time"] < fastest_time and avg_stats["avg_total_time"] > 0:
                fastest_time = avg_stats["avg_total_time"]
                fastest_model = model
        
        return {
            "models_count": len(self.test_data["models"]),
            "queries_count": len(self.test_data["queries"]),
            "total_tests": len(self.test_data["results"]),
            "fastest_model": fastest_model or "N/A",
            "model_stats": sorted(model_stats, key=lambda x: x["avg_total_time"])
        }