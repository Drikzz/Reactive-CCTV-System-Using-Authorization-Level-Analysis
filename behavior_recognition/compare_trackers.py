"""
Compare All Three Trackers on the Same Video

This script runs MobileNetV2, MobileNetV3-Small, and ShuffleNetV2 trackers on the same 
input video and generates a comprehensive performance comparison including FPS, detection 
counts, classification times, and more.

Usage:
    python compare_trackers.py --video path/to/video.mp4
    python compare_trackers.py --video path/to/video.mp4 --save-outputs
    python compare_trackers.py --webcam
"""

import argparse
import sys
import time
from pathlib import Path
import json
from typing import Dict, List
import subprocess

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))


class TrackerComparator:
    """Compare multiple tracking models on the same video."""
    
    def __init__(self, video_path: str, save_outputs: bool = False, 
                 yolo_model: str = None, conf: float = 0.5, iou: float = 0.7):
        """
        Initialize tracker comparator.
        
        Args:
            video_path: Path to video file or 'webcam'
            save_outputs: Whether to save output videos
            yolo_model: YOLOv8 model path
            conf: YOLO confidence threshold
            iou: YOLO IOU threshold
        """
        self.video_path = video_path
        self.save_outputs = save_outputs
        self.yolo_model = yolo_model or 'models/YOLOv8/yolov8m.pt'
        self.conf = conf
        self.iou = iou
        
        # Define models to compare
        self.models = {
            'MobileNetV2': {
                'script': 'behavior_recognition/MobileNetV2/yolo_mobilenet_tracker.py',
                'model_path': 'models/mobilenetv2/mobilenet_transfer.pth',
                'output': 'outputs/comparison_mobilenetv2.mp4'
            },
            'MobileNetV3-Small': {
                'script': 'behavior_recognition/mobilenetv3-small/yolo_mobilenet_tracker_mnv3small.py',
                'model_path': 'models/mobilenetv3-small/mobilenet_v3_small_transfer.pth',
                'output': 'outputs/comparison_mobilenetv3small.mp4'
            },
            'ShuffleNetV2': {
                'script': 'behavior_recognition/shufflenetv2/yolo_mobilenet_tracker_shufflenetv2.py',
                'model_path': 'models/shufflenetv2/shufflenet_v2_transfer.pth',
                'output': 'outputs/comparison_shufflenetv2.mp4'
            }
        }
        
        self.results = {}
    
    def run_tracker(self, model_name: str, model_info: Dict) -> Dict:
        """
        Run a single tracker and capture its output.
        
        Args:
            model_name: Name of the model
            model_info: Dictionary with script and model paths
            
        Returns:
            Dictionary with performance metrics
        """
        script_path = BASE_DIR / model_info['script']
        model_path = BASE_DIR / model_info['model_path']
        
        # Check if files exist
        if not script_path.exists():
            print(f"  ✗ Script not found: {script_path}")
            return {'error': 'Script not found'}
        
        if not model_path.exists():
            print(f"  ✗ Model not found: {model_path}")
            return {'error': 'Model not found'}
        
        # Build command
        cmd = [
            sys.executable,
            str(script_path),
            '--yolo-model', str(BASE_DIR / self.yolo_model),
            '--mobilenet-model', str(model_path),
            '--conf', str(self.conf),
            '--iou', str(self.iou),
            '--no-display'  # Don't display during comparison
        ]
        
        if self.video_path == 'webcam':
            cmd.append('--webcam')
        else:
            cmd.extend(['--video', self.video_path])
        
        if self.save_outputs:
            cmd.extend(['--save', model_info['output']])
        
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Running tracker...")
        
        # Run tracker and capture output
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=str(BASE_DIR),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            elapsed_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"  ✗ Error running tracker:")
                print(result.stderr)
                return {'error': 'Execution failed', 'stderr': result.stderr}
            
            # Parse output for metrics
            output = result.stdout
            metrics = self._parse_tracker_output(output)
            metrics['total_execution_time'] = elapsed_time
            
            print(f"  ✓ Completed in {elapsed_time:.2f} seconds")
            return metrics
            
        except subprocess.TimeoutExpired:
            print(f"  ✗ Timeout after 10 minutes")
            return {'error': 'Timeout'}
        except Exception as e:
            print(f"  ✗ Exception: {str(e)}")
            return {'error': str(e)}
    
    def _parse_tracker_output(self, output: str) -> Dict:
        """Parse tracker console output to extract metrics."""
        import re
        
        metrics = {}
        
        # Extract key metrics from output
        patterns = {
            'frames': r'Total Frames Processed:\s*(\d+)',
            'detections': r'Total Person Detections:\s*(\d+)',
            'classifications': r'Total Classifications:\s*(\d+)',
            'classification_rate': r'Classification Rate:\s*([\d.]+)%',
            'avg_classification_time': r'Average Classification Time:\s*([\d.]+)ms',
            'final_fps': r'Final Processing FPS:\s*([\d.]+)',
            'final_interval': r'Final Classification Interval: every\s*(\d+)\s*frames',
            'unique_tracks': r'Unique Tracks Seen:\s*(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, output)
            if match:
                try:
                    metrics[key] = float(match.group(1))
                except ValueError:
                    metrics[key] = match.group(1)
        
        return metrics
    
    def run_all(self) -> Dict[str, Dict]:
        """
        Run all trackers on the same video.
        
        Returns:
            Dictionary mapping model names to their metrics
        """
        print("\n" + "="*100)
        print("TRACKER COMPARISON")
        print("="*100)
        print(f"Video: {self.video_path}")
        print(f"YOLO Model: {self.yolo_model}")
        print(f"Save Outputs: {self.save_outputs}")
        print("="*100 + "\n")
        
        for model_name, model_info in self.models.items():
            print(f"Running {model_name}...")
            result = self.run_tracker(model_name, model_info)
            self.results[model_name] = result
            print()
        
        return self.results
    
    def generate_comparison_report(self) -> str:
        """Generate a formatted comparison report."""
        output = []
        output.append("="*100)
        output.append("TRACKER PERFORMANCE COMPARISON")
        output.append("="*100)
        output.append(f"Video: {self.video_path}")
        output.append(f"YOLO Model: {self.yolo_model}")
        output.append("")
        
        # Check for errors
        models_with_errors = [m for m, r in self.results.items() if 'error' in r]
        if models_with_errors:
            output.append(f"⚠️  Models with errors: {', '.join(models_with_errors)}")
            output.append("")
        
        # Performance metrics table
        models = [m for m in self.models.keys() if 'error' not in self.results.get(m, {})]
        
        if not models:
            output.append("❌ No successful runs to compare")
            return '\n'.join(output)
        
        output.append("PERFORMANCE METRICS")
        output.append("-"*100)
        output.append(f"{'Metric':<35} | {models[0]:<20} | {models[1]:<20} | {models[2]:<20}")
        output.append("-"*100)
        
        metrics_to_compare = [
            ('frames', 'Total Frames Processed', ''),
            ('detections', 'Total Person Detections', ''),
            ('classifications', 'Total Classifications', ''),
            ('classification_rate', 'Classification Rate', '%'),
            ('avg_classification_time', 'Avg Classification Time', 'ms'),
            ('final_fps', 'Final Processing FPS', ''),
            ('final_interval', 'Final Classification Interval', 'frames'),
            ('unique_tracks', 'Unique Tracks', ''),
            ('total_execution_time', 'Total Execution Time', 's')
        ]
        
        for key, label, unit in metrics_to_compare:
            values = []
            for model in models:
                if key in self.results[model]:
                    val = self.results[model][key]
                    if isinstance(val, float):
                        if 'time' in key.lower() and 'total' not in key.lower():
                            values.append(f"{val:.2f}{unit}")
                        elif 'fps' in key.lower():
                            values.append(f"{val:.1f}{unit}")
                        else:
                            values.append(f"{val:.1f}{unit}")
                    else:
                        values.append(f"{val}{unit}")
                else:
                    values.append("N/A")
            
            output.append(f"{label:<35} | {values[0]:<20} | {values[1]:<20} | {values[2]:<20}")
        
        output.append("-"*100)
        output.append("")
        
        # Analysis
        output.append("ANALYSIS")
        output.append("-"*100)
        
        # Find best for each metric
        fps_values = {m: self.results[m].get('final_fps', 0) for m in models}
        fastest_model = max(fps_values, key=fps_values.get) if fps_values else None
        
        class_time_values = {m: self.results[m].get('avg_classification_time', float('inf')) for m in models}
        quickest_class_model = min(class_time_values, key=class_time_values.get) if class_time_values else None
        
        exec_time_values = {m: self.results[m].get('total_execution_time', float('inf')) for m in models}
        quickest_exec_model = min(exec_time_values, key=exec_time_values.get) if exec_time_values else None
        
        if fastest_model:
            output.append(f"⚡ Fastest FPS: {fastest_model} ({fps_values[fastest_model]:.1f} FPS)")
        if quickest_class_model and class_time_values[quickest_class_model] != float('inf'):
            output.append(f"🚀 Quickest Classification: {quickest_class_model} ({class_time_values[quickest_class_model]:.2f}ms)")
        if quickest_exec_model and exec_time_values[quickest_exec_model] != float('inf'):
            output.append(f"🏃 Quickest Total Execution: {quickest_exec_model} ({exec_time_values[quickest_exec_model]:.2f}s)")
        
        output.append("")
        
        # FPS ranking
        if fps_values:
            output.append("FPS Ranking:")
            sorted_by_fps = sorted(fps_values.items(), key=lambda x: x[1], reverse=True)
            for rank, (model, fps) in enumerate(sorted_by_fps, 1):
                if fps > 0:
                    output.append(f"  {rank}. {model}: {fps:.1f} FPS")
        
        output.append("")
        
        # Speed comparison
        if len(models) >= 2 and fps_values[models[0]] > 0:
            baseline_fps = fps_values[models[0]]
            output.append(f"Speed Improvement vs {models[0]}:")
            for model in models[1:]:
                if fps_values[model] > 0:
                    improvement = ((fps_values[model] - baseline_fps) / baseline_fps) * 100
                    output.append(f"  {model}: {improvement:+.1f}%")
        
        output.append("="*100)
        
        return '\n'.join(output)
    
    def generate_markdown_report(self) -> str:
        """Generate markdown report for thesis."""
        output = []
        output.append("# Tracker Performance Comparison\n")
        output.append(f"**Video:** `{self.video_path}`  ")
        output.append(f"**YOLO Model:** `{self.yolo_model}`\n")
        
        models = [m for m in self.models.keys() if 'error' not in self.results.get(m, {})]
        
        if not models:
            output.append("❌ No successful runs to compare")
            return '\n'.join(output)
        
        # Performance table
        output.append("## Performance Metrics\n")
        output.append("| Metric | " + " | ".join(models) + " |")
        output.append("|" + "---|" * (len(models) + 1))
        
        metrics_to_compare = [
            ('frames', 'Total Frames'),
            ('detections', 'Person Detections'),
            ('classifications', 'Classifications'),
            ('classification_rate', 'Classification Rate (%)'),
            ('avg_classification_time', 'Avg Classification Time (ms)'),
            ('final_fps', '**Final Processing FPS**'),
            ('final_interval', 'Final Interval (frames)'),
            ('unique_tracks', 'Unique Tracks'),
            ('total_execution_time', 'Execution Time (s)')
        ]
        
        for key, label in metrics_to_compare:
            values = []
            for model in models:
                if key in self.results[model]:
                    val = self.results[model][key]
                    if isinstance(val, float):
                        if 'fps' in key.lower():
                            values.append(f"{val:.1f}")
                        else:
                            values.append(f"{val:.2f}")
                    else:
                        values.append(str(val))
                else:
                    values.append("N/A")
            
            output.append(f"| {label} | " + " | ".join(values) + " |")
        
        output.append("\n## Analysis\n")
        
        # Rankings
        fps_values = {m: self.results[m].get('final_fps', 0) for m in models}
        
        output.append("### FPS Ranking\n")
        sorted_by_fps = sorted(fps_values.items(), key=lambda x: x[1], reverse=True)
        for rank, (model, fps) in enumerate(sorted_by_fps, 1):
            if fps > 0:
                output.append(f"{rank}. **{model}**: {fps:.1f} FPS")
        
        output.append("\n### Speed Comparison\n")
        if len(models) >= 2 and fps_values[models[0]] > 0:
            baseline_fps = fps_values[models[0]]
            output.append(f"Relative to {models[0]}:\n")
            for model in models[1:]:
                if fps_values[model] > 0:
                    improvement = ((fps_values[model] - baseline_fps) / baseline_fps) * 100
                    output.append(f"- **{model}**: {improvement:+.1f}%")
        
        return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(
        description='Compare all three trackers on the same video'
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam (not recommended for comparison)')
    
    # Configuration
    parser.add_argument('--yolo-model', type=str,
                       default='models/YOLOv8/yolov8m.pt',
                       help='Path to YOLOv8 model (default: models/YOLOv8/yolov8m.pt)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='YOLO IOU threshold (default: 0.7)')
    
    # Output
    parser.add_argument('--save-outputs', action='store_true',
                       help='Save output videos for each model')
    parser.add_argument('--save-report', type=str, default='tracker_comparison.md',
                       help='Save comparison report (default: tracker_comparison.md)')
    parser.add_argument('--save-txt', type=str, default='tracker_comparison.txt',
                       help='Save text report (default: tracker_comparison.txt)')
    parser.add_argument('--save-json', type=str, default=None,
                       help='Save raw metrics as JSON')
    
    args = parser.parse_args()
    
    # Create comparator
    video_source = 'webcam' if args.webcam else args.video
    
    if args.webcam:
        print("⚠️  Warning: Webcam comparison may not be reproducible across runs")
        print("   Consider using a saved video file for accurate comparison\n")
    
    comparator = TrackerComparator(
        video_path=video_source,
        save_outputs=args.save_outputs,
        yolo_model=args.yolo_model,
        conf=args.conf,
        iou=args.iou
    )
    
    # Run comparison
    results = comparator.run_all()
    
    # Generate reports
    text_report = comparator.generate_comparison_report()
    markdown_report = comparator.generate_markdown_report()
    
    # Print to console
    print(text_report)
    print()
    
    # Save reports
    txt_path = Path(args.save_txt)
    with open(txt_path, 'w') as f:
        f.write(text_report)
    print(f"✓ Text report saved to: {txt_path}")
    
    md_path = Path(args.save_report)
    with open(md_path, 'w') as f:
        f.write(markdown_report)
    print(f"✓ Markdown report saved to: {md_path}")
    
    # Save JSON if requested
    if args.save_json:
        json_path = Path(args.save_json)
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ JSON metrics saved to: {json_path}")
    
    print("\n" + "="*100)
    print("COMPARISON COMPLETE")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
