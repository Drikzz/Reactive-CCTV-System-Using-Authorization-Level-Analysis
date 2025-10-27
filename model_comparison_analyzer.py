"""
Model Comparison Analyzer
Compares performance graphs from FaceNet, ArcFace, and DlibCNN analyzers
Run this after running the individual model analyzers
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import glob

# Set style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 11

class ModelComparisonAnalyzer:
    def __init__(self, base_dir="model_comparison_results"):
        self.base_dir = base_dir
        self.comparisons_dir = os.path.join(base_dir, "Comparisons")
        self.comparison_graphs_dir = os.path.join(self.comparisons_dir, "comparison_graphs")
        self.comparison_reports_dir = os.path.join(self.comparisons_dir, "comparison_reports")
        
        # Model directories
        self.model_dirs = {
            'FaceNet': os.path.join(base_dir, "FaceNet"),
            'ArcFace': os.path.join(base_dir, "ArcFace"),
            'DlibCNN': os.path.join(base_dir, "DlibCNN")
        }
        
        # Ensure directories exist
        os.makedirs(self.comparison_graphs_dir, exist_ok=True)
        os.makedirs(self.comparison_reports_dir, exist_ok=True)
        
        print(f"[INFO] Model Comparison Analyzer initialized")
        print(f"[INFO] Base directory: {self.base_dir}")
        print(f"[INFO] Comparison graphs: {self.comparison_graphs_dir}")
    
    def load_model_summaries(self):
        """Load summary data from all available models"""
        summaries = {}
        
        summaries_dir = os.path.join(self.comparisons_dir, "summaries")
        
        if not os.path.exists(summaries_dir):
            print(f"[WARN] Summaries directory not found: {summaries_dir}")
            return summaries
        
        # Look for summary files
        for model_name in ['facenet', 'arcface', 'dlib_cnn']:
            pattern = os.path.join(summaries_dir, f"{model_name}_summary_*.json")
            files = glob.glob(pattern)
            
            if files:
                # Use the most recent file
                latest_file = max(files, key=os.path.getctime)
                
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                        model_key = data.get('model', model_name.upper())
                        summaries[model_key] = data
                        print(f"[INFO] Loaded {model_key} summary from {os.path.basename(latest_file)}")
                except Exception as e:
                    print(f"[WARN] Failed to load {model_name} summary: {e}")
        
        return summaries
    
    def load_detailed_data(self):
        """Load detailed performance data from all models"""
        detailed_data = {}
        
        for model_name, model_dir in self.model_dirs.items():
            data_dir = os.path.join(model_dir, "data")
            
            if not os.path.exists(data_dir):
                continue
            
            # Look for CSV data files
            csv_pattern = os.path.join(data_dir, f"{model_name.lower()}_data_*.csv")
            csv_files = glob.glob(csv_pattern)
            
            if csv_files:
                latest_csv = max(csv_files, key=os.path.getctime)
                
                try:
                    df = pd.read_csv(latest_csv)
                    detailed_data[model_name] = df
                    print(f"[INFO] Loaded {model_name} detailed data: {len(df)} records")
                except Exception as e:
                    print(f"[WARN] Failed to load {model_name} data: {e}")
        
        return detailed_data
    
    def create_comparison_bar_chart(self, summaries):
        """Create comparison bar chart of core metrics"""
        
        if not summaries:
            print("[ERROR] No model summaries available for comparison")
            return None
        
        print(f"\n[INFO] Creating comparison bar chart...")
        
        # Extract metrics
        models = []
        confidences = []
        inference_times = []
        accuracies = []
        
        for model_name, data in summaries.items():
            metrics = data.get('metrics', {})
            
            models.append(model_name)
            confidences.append(metrics.get('avg_confidence', 0))
            inference_times.append(metrics.get('avg_inference_time_ms', 0))
            accuracies.append(metrics.get('avg_accuracy', 0))
        
        # Create figure with 3 subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        colors = ['#2E8B57', '#FF6B35', '#4169E1'][:len(models)]
        x = np.arange(len(models))
        bar_width = 0.6
        
        # 1. Average Confidence
        bars1 = ax1.bar(x, confidences, bar_width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_title('Average Confidence', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.set_ylim(0, 1.0)
        
        for bar, value in zip(bars1, confidences):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 2. Average Inference Time
        bars2 = ax2.bar(x, inference_times, bar_width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_title('Average Inference Time', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(models, fontsize=11)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, value in zip(bars2, inference_times):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(inference_times) * 0.02,
                    f'{value:.1f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # 3. Average Accuracy
        bars3 = ax3.bar(x, accuracies, bar_width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_title('Average Accuracy', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Accuracy Rate', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, fontsize=11)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1.0)
        
        for bar, value in zip(bars3, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Face Recognition Model Comparison - Core Metrics', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = os.path.join(self.comparison_graphs_dir, f"comparison_bar_chart_{timestamp}.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        
        standard_path = os.path.join(self.comparison_graphs_dir, "comparison_bar_chart_latest.png")
        plt.savefig(standard_path, dpi=300, bbox_inches='tight')
        
        print(f"[INFO] ✅ Saved comparison bar chart: {graph_path}")
        
        plt.show()
        
        return graph_path
    
    def create_radar_chart(self, summaries):
        """Create radar chart comparing normalized metrics"""
        
        if not summaries or len(summaries) < 2:
            print("[WARN] Need at least 2 models for radar chart")
            return None
        
        print(f"\n[INFO] Creating radar chart...")
        
        # Define metrics to compare (normalized to 0-1 scale)
        categories = ['Confidence', 'Speed\n(inverse time)', 'Accuracy', 'Overall\nScore']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        colors = ['#2E8B57', '#FF6B35', '#4169E1', '#9370DB', '#20B2AA']
        
        for idx, (model_name, data) in enumerate(summaries.items()):
            metrics = data.get('metrics', {})
            
            confidence = metrics.get('avg_confidence', 0)
            inference_time = metrics.get('avg_inference_time_ms', 100)
            accuracy = metrics.get('avg_accuracy', 0)
            
            # Normalize speed (lower is better, so invert)
            max_time = 200  # Assume max reasonable time
            speed_score = max(0, 1 - (inference_time / max_time))
            
            # Calculate overall score
            overall = (confidence + speed_score + accuracy) / 3
            
            values = [confidence, speed_score, accuracy, overall]
            
            # Repeat first value to close the polygon
            values += values[:1]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            angles += angles[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=colors[idx % len(colors)])
            ax.fill(angles, values, alpha=0.15, color=colors[idx % len(colors)])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
        ax.grid(True, alpha=0.3)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        plt.title('Model Performance Radar Comparison', size=16, fontweight='bold', pad=20)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = os.path.join(self.comparison_graphs_dir, f"comparison_radar_{timestamp}.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        
        standard_path = os.path.join(self.comparison_graphs_dir, "comparison_radar_latest.png")
        plt.savefig(standard_path, dpi=300, bbox_inches='tight')
        
        print(f"[INFO] ✅ Saved radar chart: {graph_path}")
        
        plt.show()
        
        return graph_path
    
    def create_detailed_comparison_table(self, summaries, detailed_data):
        """Create detailed comparison table"""
        
        if not summaries:
            print("[WARN] No summaries available for table")
            return None
        
        print(f"\n[INFO] Creating detailed comparison table...")
        
        # Prepare data for table
        table_data = []
        
        for model_name, summary in summaries.items():
            metrics = summary.get('metrics', {})
            additional = summary.get('additional_stats', {})
            
            row = {
                'Model': model_name,
                'Avg Confidence': f"{metrics.get('avg_confidence', 0):.3f}",
                'Avg Inference (ms)': f"{metrics.get('avg_inference_time_ms', 0):.1f}",
                'Avg Accuracy': f"{metrics.get('avg_accuracy', 0):.1%}",
                'Total Detections': additional.get('total_detections', 0),
                'Back View %': f"{additional.get('back_view_percentage', 0):.1f}%",
                'Locked %': f"{additional.get('locked_percentage', 0):.1f}%"
            }
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns,
                        cellLoc='center', loc='center',
                        colColours=['lightgray']*len(df.columns))
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Alternate row colors
        for i in range(1, len(df) + 1):
            for j in range(len(df.columns)):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#f0f0f0')
        
        plt.title('Detailed Model Comparison Table', fontsize=16, fontweight='bold', pad=20)
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = os.path.join(self.comparison_graphs_dir, f"comparison_table_{timestamp}.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        
        standard_path = os.path.join(self.comparison_graphs_dir, "comparison_table_latest.png")
        plt.savefig(standard_path, dpi=300, bbox_inches='tight')
        
        print(f"[INFO] ✅ Saved comparison table: {graph_path}")
        
        plt.show()
        
        # Also save as CSV
        csv_path = os.path.join(self.comparison_reports_dir, f"comparison_table_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[INFO] ✅ Saved comparison CSV: {csv_path}")
        
        return graph_path
    
    def create_performance_over_time(self, detailed_data):
        """Create performance over time comparison"""
        
        if not detailed_data or len(detailed_data) < 2:
            print("[WARN] Not enough detailed data for time series")
            return None
        
        print(f"\n[INFO] Creating performance over time comparison...")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
        
        colors = {'FaceNet': '#2E8B57', 'ArcFace': '#FF6B35', 'DlibCNN': '#4169E1'}
        
        # 1. Confidence over frames
        for model_name, df in detailed_data.items():
            if 'frame' in df.columns and 'confidence' in df.columns:
                # Group by frame and calculate mean confidence
                grouped = df[df['confidence'] > 0].groupby('frame')['confidence'].mean()
                
                ax1.plot(grouped.index, grouped.values, 
                        label=model_name, linewidth=2, alpha=0.7,
                        color=colors.get(model_name, '#333333'))
        
        ax1.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Confidence', fontsize=12, fontweight='bold')
        ax1.set_title('Confidence Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # 2. Inference time over frames
        for model_name, df in detailed_data.items():
            if 'frame' in df.columns and 'inference_time_ms' in df.columns:
                # Group by frame and calculate mean inference time
                grouped = df.groupby('frame')['inference_time_ms'].mean()
                
                ax2.plot(grouped.index, grouped.values,
                        label=model_name, linewidth=2, alpha=0.7,
                        color=colors.get(model_name, '#333333'))
        
        ax2.set_xlabel('Frame Number', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Inference Time (ms)', fontsize=12, fontweight='bold')
        ax2.set_title('Inference Time Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Model Performance Over Time Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = os.path.join(self.comparison_graphs_dir, f"comparison_timeseries_{timestamp}.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        
        standard_path = os.path.join(self.comparison_graphs_dir, "comparison_timeseries_latest.png")
        plt.savefig(standard_path, dpi=300, bbox_inches='tight')
        
        print(f"[INFO] ✅ Saved time series comparison: {graph_path}")
        
        plt.show()
        
        return graph_path
    
    def generate_comparison_report(self, summaries):
        """Generate text report comparing models"""
        
        if not summaries:
            print("[WARN] No summaries for report")
            return None
        
        print(f"\n[INFO] Generating comparison report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.comparison_reports_dir, f"comparison_report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FACE RECOGNITION MODEL COMPARISON REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Models Analyzed: {', '.join(summaries.keys())}\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 80 + "\n\n")
            
            for model_name, data in summaries.items():
                metrics = data.get('metrics', {})
                additional = data.get('additional_stats', {})
                
                f.write(f"📊 {model_name}:\n")
                f.write(f"  • Average Confidence: {metrics.get('avg_confidence', 0):.3f}\n")
                f.write(f"  • Average Inference Time: {metrics.get('avg_inference_time_ms', 0):.1f} ms\n")
                f.write(f"  • Average Accuracy: {metrics.get('avg_accuracy', 0):.1%}\n")
                f.write(f"  • Total Detections: {additional.get('total_detections', 0):,}\n")
                f.write(f"  • Back View Handling: {additional.get('back_view_percentage', 0):.1f}%\n")
                f.write(f"  • Identity Locking: {additional.get('locked_percentage', 0):.1f}%\n")
                f.write("\n")
            
            # Rankings
            f.write("\nPERFORMANCE RANKINGS\n")
            f.write("-" * 80 + "\n\n")
            
            # Confidence ranking
            conf_ranking = sorted(summaries.items(), 
                                 key=lambda x: x[1].get('metrics', {}).get('avg_confidence', 0), 
                                 reverse=True)
            f.write("🥇 Highest Confidence:\n")
            for i, (name, data) in enumerate(conf_ranking, 1):
                conf = data.get('metrics', {}).get('avg_confidence', 0)
                f.write(f"  {i}. {name}: {conf:.3f}\n")
            f.write("\n")
            
            # Speed ranking (lower is better)
            speed_ranking = sorted(summaries.items(),
                                  key=lambda x: x[1].get('metrics', {}).get('avg_inference_time_ms', 999))
            f.write("⚡ Fastest Inference:\n")
            for i, (name, data) in enumerate(speed_ranking, 1):
                time_ms = data.get('metrics', {}).get('avg_inference_time_ms', 0)
                f.write(f"  {i}. {name}: {time_ms:.1f} ms\n")
            f.write("\n")
            
            # Accuracy ranking
            acc_ranking = sorted(summaries.items(),
                                key=lambda x: x[1].get('metrics', {}).get('avg_accuracy', 0),
                                reverse=True)
            f.write("🎯 Highest Accuracy:\n")
            for i, (name, data) in enumerate(acc_ranking, 1):
                acc = data.get('metrics', {}).get('avg_accuracy', 0)
                f.write(f"  {i}. {name}: {acc:.1%}\n")
            f.write("\n")
            
            # Recommendations
            f.write("\nRECOMMENDATIONS\n")
            f.write("-" * 80 + "\n\n")
            
            best_conf = conf_ranking[0][0]
            best_speed = speed_ranking[0][0]
            best_acc = acc_ranking[0][0]
            
            f.write(f"• For HIGHEST CONFIDENCE: Use {best_conf}\n")
            f.write(f"• For FASTEST PROCESSING: Use {best_speed}\n")
            f.write(f"• For BEST ACCURACY: Use {best_acc}\n")
            f.write("\n")
            
            # Calculate overall best
            scores = {}
            for name, data in summaries.items():
                metrics = data.get('metrics', {})
                conf = metrics.get('avg_confidence', 0)
                time_ms = metrics.get('avg_inference_time_ms', 100)
                acc = metrics.get('avg_accuracy', 0)
                
                # Normalize and combine (speed inverted)
                speed_score = max(0, 1 - (time_ms / 200))
                overall = (conf + speed_score + acc) / 3
                scores[name] = overall
            
            best_overall = max(scores.items(), key=lambda x: x[1])
            f.write(f"• OVERALL BEST BALANCED: {best_overall[0]} (score: {best_overall[1]:.3f})\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"[INFO] ✅ Saved comparison report: {report_path}")
        
        # Also print to console
        with open(report_path, 'r') as f:
            print(f"\n{f.read()}")
        
        return report_path
    
    def run_full_comparison(self):
        """Run complete comparison analysis"""
        
        print("\n" + "=" * 80)
        print("🔬 STARTING COMPREHENSIVE MODEL COMPARISON")
        print("=" * 80 + "\n")
        
        # Load data
        summaries = self.load_model_summaries()
        detailed_data = self.load_detailed_data()
        
        if not summaries:
            print("[ERROR] No model summaries found. Run individual analyzers first!")
            print("[INFO] Run these scripts first:")
            print("  1. python facenet_video_analyzer.py")
            print("  2. python arcface_video_analyzer.py")
            print("  3. python dlib_cnn_video_analyzer.py")
            return False
        
        print(f"\n[INFO] Found {len(summaries)} models to compare: {list(summaries.keys())}")
        
        # Generate all comparisons
        graphs_created = []
        
        # 1. Bar chart
        bar_path = self.create_comparison_bar_chart(summaries)
        if bar_path:
            graphs_created.append(bar_path)
        
        # 2. Radar chart
        radar_path = self.create_radar_chart(summaries)
        if radar_path:
            graphs_created.append(radar_path)
        
        # 3. Comparison table
        table_path = self.create_detailed_comparison_table(summaries, detailed_data)
        if table_path:
            graphs_created.append(table_path)
        
        # 4. Time series (if detailed data available)
        if detailed_data:
            time_path = self.create_performance_over_time(detailed_data)
            if time_path:
                graphs_created.append(time_path)
        
        # 5. Text report
        report_path = self.generate_comparison_report(summaries)
        
        # Summary
        print("\n" + "=" * 80)
        print("✅ COMPARISON COMPLETE!")
        print("=" * 80)
        print(f"\n📊 Generated {len(graphs_created)} comparison graphs:")
        for path in graphs_created:
            print(f"  • {os.path.basename(path)}")
        
        if report_path:
            print(f"\n📄 Generated report: {os.path.basename(report_path)}")
        
        print(f"\n📁 All outputs saved to:")
        print(f"  Graphs: {self.comparison_graphs_dir}")
        print(f"  Reports: {self.comparison_reports_dir}")
        
        print("\n" + "=" * 80)
        
        return True

def main():
    """Main comparison function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare face recognition model performances")
    parser.add_argument("--base-dir", "-d",
                       default="model_comparison_results",
                       help="Base directory containing model results")
    
    args = parser.parse_args()
    
    print("🔬 MODEL COMPARISON ANALYZER")
    print("=" * 80)
    print("[INFO] This tool compares results from FaceNet, ArcFace, and DlibCNN analyzers")
    print("[INFO] Make sure you have run the individual model analyzers first!")
    print("=" * 80)
    
    analyzer = ModelComparisonAnalyzer(args.base_dir)
    
    if not analyzer.run_full_comparison():
        print("\n[ERROR] Comparison failed. Please check that model analyzers have been run.")
        return 1
    
    print("\n💡 TIP: You can re-run this script anytime to regenerate comparisons")
    print("💡 TIP: Individual model graphs are preserved in their respective folders")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())