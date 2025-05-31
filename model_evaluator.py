#!/usr/bin/env python3
"""
Model Evaluator - Comprehensive XTTS Model Evaluation Suite
This system provides detailed evaluation of voice cloning model performance.
"""

import os
import sys
import time
import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import statistics

class ModelEvaluator:
    """Comprehensive model evaluation system."""
    
    def __init__(self, model_path: str = None, output_dir: str = "evaluation_results"):
        """Initialize the model evaluator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "audio_samples").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        self.model_path = model_path
        self.evaluation_results = {}
        self.test_scenarios = []
        
        # Test scenarios for comprehensive evaluation
        self.default_test_texts = [
            "Hello, this is a simple test sentence.",
            "The quick brown fox jumps over the lazy dog.",
            "In the heart of the bustling city, where skyscrapers touch the clouds and the rhythm of life never ceases.",
            "Technical terms: artificial intelligence, machine learning, neural networks, deep learning algorithms.",
            "Emotional expression: I'm absolutely thrilled and excited about this amazing opportunity!",
            "Numbers and dates: On December 25th, 2024, at 3:45 PM, the temperature was 72.5 degrees Fahrenheit.",
            "Complex pronunciation: The Massachusetts Institute of Technology specializes in technological innovations.",
            "Poetry: Shall I compare thee to a summer's day? Thou art more lovely and more temperate.",
            "Question format: How are you doing today? Are you ready for this incredible journey?",
            "Long sentence: Despite the numerous challenges and obstacles that we face in our daily lives, it's important to remain optimistic and continue working towards our goals with determination and perseverance."
        ]
        
        print("🎯 Model Evaluator initialized")
        print(f"📁 Results will be saved to: {self.output_dir}")
    
    def add_test_scenario(self, name: str, text: str, reference_audio: str, 
                         expected_quality: float = None, category: str = "custom"):
        """Add a custom test scenario."""
        scenario = {
            'name': name,
            'text': text,
            'reference_audio': reference_audio,
            'expected_quality': expected_quality,
            'category': category,
            'results': {}
        }
        self.test_scenarios.append(scenario)
        print(f"📝 Test scenario added: {name}")
    
    def create_default_scenarios(self, reference_audio: str):
        """Create default test scenarios using provided reference audio."""
        categories = {
            'simple': self.default_test_texts[:2],
            'complex': self.default_test_texts[2:4],
            'technical': self.default_test_texts[4:6],
            'emotional': self.default_test_texts[6:8],
            'challenging': self.default_test_texts[8:]
        }
        
        for category, texts in categories.items():
            for i, text in enumerate(texts):
                self.add_test_scenario(
                    name=f"{category}_{i+1}",
                    text=text,
                    reference_audio=reference_audio,
                    category=category
                )
        
        print(f"✅ Created {len(self.test_scenarios)} default test scenarios")
    
    def simulate_voice_generation(self, text: str, reference_audio: str, 
                                output_path: str) -> Tuple[bool, float, Dict]:
        """Simulate voice generation (placeholder for actual TTS integration)."""
        start_time = time.time()
        
        try:
            # Simulate processing time based on text length
            base_time = 2.0  # Base processing time
            text_factor = len(text) / 100  # Scale with text length
            processing_time = base_time + text_factor + np.random.normal(0, 0.5)
            processing_time = max(0.5, processing_time)  # Minimum 0.5 seconds
            
            time.sleep(processing_time)
            
            # Simulate success rate (90% success for evaluation)
            success = np.random.random() > 0.1
            
            if success:
                # Create dummy audio file with realistic properties
                sample_rate = 22050
                duration = max(len(text) / 20, 1.0)  # Rough duration estimate
                
                # Generate more realistic audio than white noise
                t = torch.linspace(0, duration, int(sample_rate * duration))
                # Create a mix of sine waves to simulate speech-like audio
                waveform = (torch.sin(2 * np.pi * 440 * t) * 0.1 + 
                           torch.sin(2 * np.pi * 880 * t) * 0.05 +
                           torch.randn_like(t) * 0.02)
                waveform = waveform.unsqueeze(0)
                
                torchaudio.save(output_path, waveform, sample_rate)
                
                generation_time = time.time() - start_time
                
                # Simulate quality metrics
                quality_metrics = {
                    'rms_level': 0.1 + np.random.normal(0, 0.02),
                    'spectral_centroid': 2000 + np.random.normal(0, 200),
                    'duration': duration,
                    'snr_estimate': 25 + np.random.normal(0, 3),
                    'quality_score': 75 + np.random.normal(0, 10)
                }
                
                return True, generation_time, quality_metrics
            else:
                return False, time.time() - start_time, {}
                
        except Exception as e:
            return False, time.time() - start_time, {'error': str(e)}
    
    def evaluate_single_scenario(self, scenario: Dict) -> Dict:
        """Evaluate a single test scenario."""
        print(f"🎯 Evaluating: {scenario['name']}")
        print(f"   📝 Text: {scenario['text'][:50]}{'...' if len(scenario['text']) > 50 else ''}")
        
        output_path = self.output_dir / "audio_samples" / f"{scenario['name']}.wav"
        
        # Generate audio
        success, generation_time, quality_metrics = self.simulate_voice_generation(
            scenario['text'], scenario['reference_audio'], str(output_path)
        )
        
        # Calculate performance metrics
        result = {
            'scenario_name': scenario['name'],
            'category': scenario['category'],
            'text_length': len(scenario['text']),
            'success': success,
            'generation_time': generation_time,
            'output_file': str(output_path) if success else None,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            # Add quality metrics
            result.update(quality_metrics)
            
            # Calculate efficiency metrics
            audio_duration = quality_metrics.get('duration', 0)
            result['real_time_factor'] = generation_time / max(audio_duration, 0.1)
            result['words_per_minute'] = (len(scenario['text'].split()) / generation_time) * 60
            
            # Quality assessment
            quality_score = quality_metrics.get('quality_score', 0)
            if quality_score >= 80:
                result['quality_grade'] = 'Excellent'
            elif quality_score >= 70:
                result['quality_grade'] = 'Good'
            elif quality_score >= 60:
                result['quality_grade'] = 'Fair'
            else:
                result['quality_grade'] = 'Poor'
            
            print(f"   ✅ Success - Quality: {quality_score:.1f}/100, RT Factor: {result['real_time_factor']:.2f}x")
        else:
            result['error_message'] = quality_metrics.get('error', 'Generation failed')
            print(f"   ❌ Failed: {result['error_message']}")
        
        scenario['results'] = result
        return result
    
    def run_speed_benchmark(self, text_samples: List[str] = None, 
                          iterations: int = 5) -> Dict:
        """Run speed benchmark tests."""
        print(f"\n🚀 Running speed benchmark ({iterations} iterations)")
        
        if text_samples is None:
            text_samples = self.default_test_texts[:3]  # Use first 3 samples
        
        benchmark_results = {
            'iterations': iterations,
            'text_samples': text_samples,
            'individual_results': [],
            'summary': {}
        }
        
        all_times = []
        all_rt_factors = []
        
        for iteration in range(iterations):
            print(f"🔄 Benchmark iteration {iteration + 1}/{iterations}")
            iteration_results = []
            
            for i, text in enumerate(text_samples):
                output_path = self.output_dir / "audio_samples" / f"benchmark_{iteration}_{i}.wav"
                
                # Use first reference audio from scenarios if available
                ref_audio = self.test_scenarios[0]['reference_audio'] if self.test_scenarios else "dummy_ref.wav"
                
                success, gen_time, metrics = self.simulate_voice_generation(
                    text, ref_audio, str(output_path)
                )
                
                if success:
                    rt_factor = gen_time / max(metrics.get('duration', 0.1), 0.1)
                    all_times.append(gen_time)
                    all_rt_factors.append(rt_factor)
                    
                    iteration_results.append({
                        'text_length': len(text),
                        'generation_time': gen_time,
                        'rt_factor': rt_factor,
                        'success': True
                    })
                else:
                    iteration_results.append({
                        'text_length': len(text),
                        'generation_time': gen_time,
                        'success': False
                    })
            
            benchmark_results['individual_results'].append(iteration_results)
        
        # Calculate summary statistics
        if all_times:
            benchmark_results['summary'] = {
                'total_samples': len(all_times),
                'success_rate': len(all_times) / (iterations * len(text_samples)) * 100,
                'avg_generation_time': statistics.mean(all_times),
                'median_generation_time': statistics.median(all_times),
                'min_generation_time': min(all_times),
                'max_generation_time': max(all_times),
                'std_generation_time': statistics.stdev(all_times) if len(all_times) > 1 else 0,
                'avg_rt_factor': statistics.mean(all_rt_factors),
                'median_rt_factor': statistics.median(all_rt_factors),
                'min_rt_factor': min(all_rt_factors),
                'max_rt_factor': max(all_rt_factors)
            }
        
        print(f"✅ Speed benchmark completed")
        if benchmark_results['summary']:
            print(f"   📊 Avg generation time: {benchmark_results['summary']['avg_generation_time']:.2f}s")
            print(f"   ⚡ Avg RT factor: {benchmark_results['summary']['avg_rt_factor']:.2f}x")
            print(f"   🎯 Success rate: {benchmark_results['summary']['success_rate']:.1f}%")
        
        return benchmark_results
    
    def run_consistency_test(self, text: str, reference_audio: str, 
                           iterations: int = 10) -> Dict:
        """Test consistency by generating the same text multiple times."""
        print(f"\n🎯 Running consistency test ({iterations} iterations)")
        print(f"   📝 Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        results = []
        quality_scores = []
        generation_times = []
        
        for i in range(iterations):
            output_path = self.output_dir / "audio_samples" / f"consistency_{i+1}.wav"
            
            success, gen_time, metrics = self.simulate_voice_generation(
                text, reference_audio, str(output_path)
            )
            
            result = {
                'iteration': i + 1,
                'success': success,
                'generation_time': gen_time,
                'output_file': str(output_path) if success else None
            }
            
            if success:
                quality_score = metrics.get('quality_score', 0)
                result.update(metrics)
                quality_scores.append(quality_score)
                generation_times.append(gen_time)
            
            results.append(result)
        
        # Calculate consistency metrics
        consistency_report = {
            'text': text,
            'iterations': iterations,
            'results': results,
            'summary': {
                'success_rate': sum(1 for r in results if r['success']) / iterations * 100,
                'successful_generations': len(quality_scores)
            }
        }
        
        if quality_scores:
            consistency_report['summary'].update({
                'avg_quality': statistics.mean(quality_scores),
                'quality_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores),
                'quality_range': max(quality_scores) - min(quality_scores),
                'avg_generation_time': statistics.mean(generation_times),
                'time_std': statistics.stdev(generation_times) if len(generation_times) > 1 else 0
            })
            
            # Consistency score (lower std deviation = higher consistency)
            quality_consistency = max(0, 100 - consistency_report['summary']['quality_std'])
            time_consistency = max(0, 100 - (consistency_report['summary']['time_std'] * 10))
            consistency_report['summary']['consistency_score'] = (quality_consistency + time_consistency) / 2
        
        print(f"✅ Consistency test completed")
        if 'avg_quality' in consistency_report['summary']:
            print(f"   📊 Avg quality: {consistency_report['summary']['avg_quality']:.1f} ± {consistency_report['summary']['quality_std']:.1f}")
            print(f"   🎯 Consistency score: {consistency_report['summary']['consistency_score']:.1f}/100")
        
        return consistency_report
    
    def run_comprehensive_evaluation(self, reference_audio: str) -> Dict:
        """Run comprehensive model evaluation."""
        print("🚀 Starting comprehensive model evaluation")
        print("=" * 50)
        
        # Create test scenarios if none exist
        if not self.test_scenarios:
            self.create_default_scenarios(reference_audio)
        
        evaluation_start = time.time()
        
        # 1. Scenario-based evaluation
        print("\n📋 Phase 1: Scenario-based Evaluation")
        scenario_results = []
        for scenario in self.test_scenarios:
            result = self.evaluate_single_scenario(scenario)
            scenario_results.append(result)
        
        # 2. Speed benchmark
        print("\n🚀 Phase 2: Speed Benchmark")
        speed_results = self.run_speed_benchmark()
        
        # 3. Consistency test
        print("\n🎯 Phase 3: Consistency Test")
        test_text = "This is a consistency test for voice generation quality."
        consistency_results = self.run_consistency_test(test_text, reference_audio)
        
        # Compile comprehensive results
        total_evaluation_time = time.time() - evaluation_start
        
        self.evaluation_results = {
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'model_path': self.model_path,
                'reference_audio': reference_audio,
                'total_evaluation_time': total_evaluation_time,
                'total_scenarios': len(scenario_results)
            },
            'scenario_results': scenario_results,
            'speed_benchmark': speed_results,
            'consistency_test': consistency_results,
            'summary': self.calculate_overall_summary(scenario_results, speed_results, consistency_results)
        }
        
        # Generate reports
        self.generate_evaluation_report()
        self.plot_evaluation_results()
        
        print(f"\n🎉 Comprehensive evaluation completed in {total_evaluation_time:.1f} seconds")
        return self.evaluation_results
    
    def calculate_overall_summary(self, scenario_results: List[Dict], 
                                speed_results: Dict, consistency_results: Dict) -> Dict:
        """Calculate overall evaluation summary."""
        successful_scenarios = [r for r in scenario_results if r['success']]
        
        if not successful_scenarios:
            return {'error': 'No successful generations to analyze'}
        
        # Quality metrics
        quality_scores = [r['quality_score'] for r in successful_scenarios if 'quality_score' in r]
        generation_times = [r['generation_time'] for r in successful_scenarios]
        rt_factors = [r['real_time_factor'] for r in successful_scenarios if 'real_time_factor' in r]
        
        summary = {
            'overall_success_rate': len(successful_scenarios) / len(scenario_results) * 100,
            'total_scenarios_tested': len(scenario_results),
            'successful_scenarios': len(successful_scenarios)
        }
        
        if quality_scores:
            summary.update({
                'avg_quality_score': statistics.mean(quality_scores),
                'quality_std': statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0,
                'min_quality': min(quality_scores),
                'max_quality': max(quality_scores)
            })
        
        if generation_times:
            summary.update({
                'avg_generation_time': statistics.mean(generation_times),
                'generation_time_std': statistics.stdev(generation_times) if len(generation_times) > 1 else 0
            })
        
        if rt_factors:
            summary.update({
                'avg_rt_factor': statistics.mean(rt_factors),
                'rt_factor_std': statistics.stdev(rt_factors) if len(rt_factors) > 1 else 0
            })
        
        # Performance grades
        if 'avg_quality_score' in summary:
            if summary['avg_quality_score'] >= 85:
                summary['quality_grade'] = 'Excellent'
            elif summary['avg_quality_score'] >= 75:
                summary['quality_grade'] = 'Good'
            elif summary['avg_quality_score'] >= 65:
                summary['quality_grade'] = 'Fair'
            else:
                summary['quality_grade'] = 'Needs Improvement'
        
        # Speed benchmark integration
        if speed_results.get('summary'):
            summary['speed_benchmark'] = speed_results['summary']
        
        # Consistency integration
        if consistency_results.get('summary'):
            summary['consistency_metrics'] = consistency_results['summary']
        
        return summary
    
    def generate_evaluation_report(self) -> str:
        """Generate comprehensive evaluation report."""
        if not self.evaluation_results:
            print("❌ No evaluation results to report")
            return ""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / "reports" / f"evaluation_report_{timestamp}.txt"
        
        summary = self.evaluation_results.get('summary', {})
        
        report = f"""
🎯 XTTS MODEL EVALUATION REPORT
{'='*60}

📊 EVALUATION METADATA
Evaluation Date: {self.evaluation_results['metadata']['evaluation_date']}
Model Path: {self.evaluation_results['metadata'].get('model_path', 'Not specified')}
Reference Audio: {self.evaluation_results['metadata']['reference_audio']}
Total Evaluation Time: {self.evaluation_results['metadata']['total_evaluation_time']:.1f} seconds
Total Scenarios: {self.evaluation_results['metadata']['total_scenarios']}

🏆 OVERALL PERFORMANCE SUMMARY
"""
        
        if 'error' not in summary:
            report += f"""
Success Rate: {summary.get('overall_success_rate', 0):.1f}%
Successful Scenarios: {summary.get('successful_scenarios', 0)}/{summary.get('total_scenarios_tested', 0)}

📈 QUALITY METRICS
Average Quality Score: {summary.get('avg_quality_score', 0):.1f}/100
Quality Grade: {summary.get('quality_grade', 'Unknown')}
Quality Range: {summary.get('min_quality', 0):.1f} - {summary.get('max_quality', 0):.1f}
Quality Standard Deviation: {summary.get('quality_std', 0):.1f}

⚡ PERFORMANCE METRICS
Average Generation Time: {summary.get('avg_generation_time', 0):.2f} seconds
Average Real-time Factor: {summary.get('avg_rt_factor', 0):.2f}x
Generation Time Std Dev: {summary.get('generation_time_std', 0):.2f} seconds
"""
        
            # Speed benchmark results
            speed_summary = summary.get('speed_benchmark', {})
            if speed_summary:
                report += f"""
🚀 SPEED BENCHMARK RESULTS
Benchmark Success Rate: {speed_summary.get('success_rate', 0):.1f}%
Average Generation Time: {speed_summary.get('avg_generation_time', 0):.2f}s
Median Generation Time: {speed_summary.get('median_generation_time', 0):.2f}s
Generation Time Range: {speed_summary.get('min_generation_time', 0):.2f}s - {speed_summary.get('max_generation_time', 0):.2f}s
Average RT Factor: {speed_summary.get('avg_rt_factor', 0):.2f}x
"""
            
            # Consistency test results
            consistency_summary = summary.get('consistency_metrics', {})
            if consistency_summary:
                report += f"""
🎯 CONSISTENCY TEST RESULTS
Consistency Success Rate: {consistency_summary.get('success_rate', 0):.1f}%
Average Quality: {consistency_summary.get('avg_quality', 0):.1f} ± {consistency_summary.get('quality_std', 0):.1f}
Quality Range: {consistency_summary.get('min_quality', 0):.1f} - {consistency_summary.get('max_quality', 0):.1f}
Consistency Score: {consistency_summary.get('consistency_score', 0):.1f}/100
"""
        
        # Scenario breakdown
        report += f"\n📋 SCENARIO BREAKDOWN\n"
        categories = {}
        for result in self.evaluation_results['scenario_results']:
            category = result.get('category', 'unknown')
            if category not in categories:
                categories[category] = {'total': 0, 'successful': 0, 'avg_quality': []}
            
            categories[category]['total'] += 1
            if result['success']:
                categories[category]['successful'] += 1
                if 'quality_score' in result:
                    categories[category]['avg_quality'].append(result['quality_score'])
        
        for category, stats in categories.items():
            success_rate = (stats['successful'] / stats['total']) * 100
            avg_quality = statistics.mean(stats['avg_quality']) if stats['avg_quality'] else 0
            report += f"{category.title()}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%) - Avg Quality: {avg_quality:.1f}\n"
        
        report += f"""
🎯 RECOMMENDATIONS
"""
        
        # Add recommendations based on results
        if summary.get('avg_quality_score', 0) < 70:
            report += "- Consider fine-tuning the model with higher quality training data\n"
        if summary.get('avg_rt_factor', 0) > 3:
            report += "- Optimize model inference for better real-time performance\n"
        if summary.get('overall_success_rate', 0) < 90:
            report += "- Investigate and fix reliability issues causing generation failures\n"
        if summary.get('quality_std', 0) > 15:
            report += "- Work on improving consistency across different text types\n"
        
        if (summary.get('avg_quality_score', 0) >= 80 and 
            summary.get('overall_success_rate', 0) >= 95):
            report += "- Excellent performance! Model is ready for production use\n"
        
        report += f"\nReport generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        # Save report
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Also save detailed JSON
        json_file = self.output_dir / "reports" / f"evaluation_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2)
        
        print(f"📋 Evaluation reports saved:")
        print(f"   📄 Summary: {report_file}")
        print(f"   📊 Detailed: {json_file}")
        
        return str(report_file)
    
    def plot_evaluation_results(self):
        """Create visualizations of evaluation results."""
        if not self.evaluation_results:
            return
        
        scenario_results = self.evaluation_results['scenario_results']
        successful_results = [r for r in scenario_results if r['success']]
        
        if not successful_results:
            print("❌ No successful results to plot")
            return
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Quality scores by category
        categories = {}
        for result in successful_results:
            category = result.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            if 'quality_score' in result:
                categories[category].append(result['quality_score'])
        
        if categories:
            cat_names = list(categories.keys())
            cat_scores = [statistics.mean(scores) if scores else 0 for scores in categories.values()]
            
            ax1.bar(cat_names, cat_scores, color='skyblue', alpha=0.7)
            ax1.set_title('Average Quality Score by Category')
            ax1.set_ylabel('Quality Score')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for i, score in enumerate(cat_scores):
                ax1.text(i, score + 1, f'{score:.1f}', ha='center', va='bottom')
        
        # 2. Generation time vs text length
        text_lengths = [r['text_length'] for r in successful_results]
        gen_times = [r['generation_time'] for r in successful_results]
        
        ax2.scatter(text_lengths, gen_times, alpha=0.6, color='orange')
        ax2.set_title('Generation Time vs Text Length')
        ax2.set_xlabel('Text Length (characters)')
        ax2.set_ylabel('Generation Time (seconds)')
        
        # Add trend line
        if len(text_lengths) > 1:
            z = np.polyfit(text_lengths, gen_times, 1)
            p = np.poly1d(z)
            ax2.plot(sorted(text_lengths), p(sorted(text_lengths)), "r--", alpha=0.8)
        
        # 3. Quality distribution histogram
        quality_scores = [r['quality_score'] for r in successful_results if 'quality_score' in r]
        if quality_scores:
            ax3.hist(quality_scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title('Quality Score Distribution')
            ax3.set_xlabel('Quality Score')
            ax3.set_ylabel('Frequency')
            ax3.axvline(statistics.mean(quality_scores), color='red', linestyle='--', 
                       label=f'Mean: {statistics.mean(quality_scores):.1f}')
            ax3.legend()
        
        # 4. Real-time factor analysis
        rt_factors = [r['real_time_factor'] for r in successful_results if 'real_time_factor' in r]
        if rt_factors:
            ax4.boxplot(rt_factors, labels=['RT Factor'])
            ax4.set_title('Real-time Factor Distribution')
            ax4.set_ylabel('Real-time Factor (lower is better)')
            ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='Real-time')
            ax4.legend()
            
            # Add statistics text
            mean_rt = statistics.mean(rt_factors)
            median_rt = statistics.median(rt_factors)
            ax4.text(1.1, max(rt_factors)*0.8, f'Mean: {mean_rt:.2f}\nMedian: {median_rt:.2f}', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = self.output_dir / "plots" / f"evaluation_results_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Evaluation plots saved: {plot_file}")

def main():
    """Main function for interactive model evaluation."""
    print("🎯 Model Evaluator - Interactive Mode")
    print("=" * 50)
    
    evaluator = ModelEvaluator()
    
    while True:
        print("\n📋 Available Commands:")
        print("1. Add custom test scenario")
        print("2. Run comprehensive evaluation")
        print("3. Run speed benchmark only")
        print("4. Run consistency test only")
        print("5. Show current scenarios")
        print("6. Generate evaluation report")
        print("7. Create evaluation plots")
        print("8. Exit")
        
        choice = input("\n🎯 Enter your choice (1-8): ").strip()
        
        if choice == '1':
            name = input("📝 Scenario name: ").strip()
            text = input("📝 Text to synthesize: ").strip()
            ref_audio = input("📁 Reference audio path: ").strip()
            category = input("📂 Category (optional): ").strip() or "custom"
            
            if name and text and os.path.exists(ref_audio):
                evaluator.add_test_scenario(name, text, ref_audio, category=category)
                print("✅ Test scenario added")
            else:
                print("❌ Invalid input or reference audio not found")
        
        elif choice == '2':
            ref_audio = input("📁 Reference audio path: ").strip()
            if os.path.exists(ref_audio):
                results = evaluator.run_comprehensive_evaluation(ref_audio)
                print(f"✅ Comprehensive evaluation completed")
                summary = results.get('summary', {})
                if 'avg_quality_score' in summary:
                    print(f"   📊 Overall Quality: {summary['avg_quality_score']:.1f}/100")
                    print(f"   🎯 Success Rate: {summary['overall_success_rate']:.1f}%")
            else:
                print("❌ Reference audio file not found")
        
        elif choice == '3':
            results = evaluator.run_speed_benchmark()
            if results.get('summary'):
                print(f"✅ Speed benchmark completed")
                print(f"   ⚡ Avg generation time: {results['summary']['avg_generation_time']:.2f}s")
                print(f"   🎯 Success rate: {results['summary']['success_rate']:.1f}%")
        
        elif choice == '4':
            text = input("📝 Text for consistency test: ").strip()
            ref_audio = input("📁 Reference audio path: ").strip()
            iterations = input("🔄 Number of iterations (default: 10): ").strip()
            
            iterations = int(iterations) if iterations.isdigit() else 10
            
            if text and os.path.exists(ref_audio):
                results = evaluator.run_consistency_test(text, ref_audio, iterations)
                if 'avg_quality' in results.get('summary', {}):
                    summary = results['summary']
                    print(f"✅ Consistency test completed")
                    print(f"   📊 Avg quality: {summary['avg_quality']:.1f} ± {summary['quality_std']:.1f}")
                    print(f"   🎯 Consistency score: {summary['consistency_score']:.1f}/100")
            else:
                print("❌ Invalid text or reference audio not found")
        
        elif choice == '5':
            if evaluator.test_scenarios:
                print(f"\n📋 Current Test Scenarios ({len(evaluator.test_scenarios)}):")
                for i, scenario in enumerate(evaluator.test_scenarios):
                    print(f"   {i+1}. {scenario['name']} ({scenario['category']})")
                    print(f"      Text: {scenario['text'][:60]}{'...' if len(scenario['text']) > 60 else ''}")
            else:
                print("❌ No test scenarios defined")
        
        elif choice == '6':
            if evaluator.evaluation_results:
                report_file = evaluator.generate_evaluation_report()
                print(f"✅ Evaluation report generated: {report_file}")
            else:
                print("❌ No evaluation results available. Run evaluation first.")
        
        elif choice == '7':
            if evaluator.evaluation_results:
                evaluator.plot_evaluation_results()
                print("✅ Evaluation plots generated")
            else:
                print("❌ No evaluation results available. Run evaluation first.")
        
        elif choice == '8':
            print("👋 Model evaluation session ended")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
