#!/usr/bin/env python3
"""
Batch Voice Processor - Efficient Batch Voice Generation System
This system enables batch processing of voice generation tasks with configuration management.
"""

import os
import sys
import json
import time
import torch
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid

class BatchVoiceProcessor:
    """Batch processing system for voice generation tasks."""
    
    def __init__(self, config_file: Optional[str] = None, output_dir: str = "batch_output"):
        """Initialize the batch processor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "manifests").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        self.config = self.load_config(config_file) if config_file else self.get_default_config()
        self.job_queue = []
        self.completed_jobs = []
        self.failed_jobs = []
        self.processing_stats = {
            'start_time': None,
            'end_time': None,
            'total_jobs': 0,
            'completed': 0,
            'failed': 0,
            'processing_time': 0
        }
        
        print("🚀 Batch Voice Processor initialized")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"⚙️  Configuration loaded with {len(self.config.get('jobs', []))} jobs")
    
    def get_default_config(self) -> Dict:
        """Get default batch processing configuration."""
        return {
            "batch_settings": {
                "max_workers": 2,
                "timeout_per_job": 300,
                "retry_failed": True,
                "max_retries": 2,
                "save_intermediate": True
            },
            "model_settings": {
                "model_path": "output_fixed/run/training/GPT_XTTS_FT-*/best_model.pth",
                "config_path": "output_fixed/run/training/GPT_XTTS_FT-*/config.json",
                "language": "en",
                "speaker_wav": "data/dieck/dataset/dieck_001.wav"
            },
            "output_settings": {
                "audio_format": "wav",
                "sample_rate": 22050,
                "generate_manifest": True,
                "generate_report": True
            },
            "jobs": []
        }
    
    def load_config(self, config_file: str) -> Dict:
        """Load batch configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"✅ Configuration loaded from {config_file}")
            return config
        except Exception as e:
            print(f"❌ Error loading config {config_file}: {e}")
            print("🔄 Using default configuration")
            return self.get_default_config()
    
    def save_config(self, config_file: str):
        """Save current configuration to JSON file."""
        try:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            print(f"✅ Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def add_job(self, text: str, reference_audio: str = None, 
                output_name: str = None, job_id: str = None) -> str:
        """Add a new job to the processing queue."""
        if job_id is None:
            job_id = str(uuid.uuid4())[:8]
        
        if output_name is None:
            output_name = f"batch_output_{job_id}.wav"
        
        job = {
            'job_id': job_id,
            'text': text,
            'reference_audio': reference_audio or self.config['model_settings']['speaker_wav'],
            'output_name': output_name,
            'output_path': str(self.output_dir / "audio" / output_name),
            'status': 'queued',
            'created_at': datetime.now().isoformat(),
            'attempts': 0,
            'error_message': None
        }
        
        self.job_queue.append(job)
        print(f"📝 Job added: {job_id} - '{text[:50]}{'...' if len(text) > 50 else ''}'")
        return job_id
    
    def add_jobs_from_list(self, text_list: List[str], reference_audio: str = None) -> List[str]:
        """Add multiple jobs from a list of texts."""
        job_ids = []
        for i, text in enumerate(text_list):
            job_id = self.add_job(
                text=text,
                reference_audio=reference_audio,
                output_name=f"batch_job_{i+1:03d}.wav"
            )
            job_ids.append(job_id)
        
        print(f"✅ Added {len(job_ids)} jobs to queue")
        return job_ids
    
    def load_jobs_from_file(self, jobs_file: str) -> int:
        """Load jobs from a text file (one job per line) or JSON file."""
        try:
            if jobs_file.endswith('.json'):
                with open(jobs_file, 'r') as f:
                    data = json.load(f)
                    if 'jobs' in data:
                        for job_data in data['jobs']:
                            self.add_job(
                                text=job_data['text'],
                                reference_audio=job_data.get('reference_audio'),
                                output_name=job_data.get('output_name'),
                                job_id=job_data.get('job_id')
                            )
                        return len(data['jobs'])
            else:
                with open(jobs_file, 'r') as f:
                    texts = [line.strip() for line in f if line.strip()]
                    self.add_jobs_from_list(texts)
                    return len(texts)
        except Exception as e:
            print(f"❌ Error loading jobs from {jobs_file}: {e}")
            return 0
    
    def simulate_voice_generation(self, job: Dict) -> Tuple[bool, str]:
        """Simulate voice generation for a job (placeholder for actual TTS integration)."""
        try:
            print(f"🎙️  Processing job {job['job_id']}: '{job['text'][:30]}...'")
            
            # Simulate processing time
            processing_time = np.random.uniform(2, 8)  # 2-8 seconds per job
            time.sleep(processing_time)
            
            # Simulate success/failure (95% success rate)
            if np.random.random() > 0.05:
                # Create dummy audio file for demonstration
                sample_rate = self.config['output_settings']['sample_rate']
                duration = max(len(job['text']) / 20, 1.0)  # Rough duration estimate
                
                # Generate white noise as placeholder
                waveform = torch.randn(1, int(sample_rate * duration)) * 0.1
                torchaudio.save(job['output_path'], waveform, sample_rate)
                
                return True, f"Generated in {processing_time:.1f}s"
            else:
                return False, "Simulated generation failure"
                
        except Exception as e:
            return False, f"Error during generation: {e}"
    
    def process_single_job(self, job: Dict) -> Dict:
        """Process a single job."""
        job['attempts'] += 1
        job['started_at'] = datetime.now().isoformat()
        job['status'] = 'processing'
        
        success, message = self.simulate_voice_generation(job)
        
        job['completed_at'] = datetime.now().isoformat()
        
        if success:
            job['status'] = 'completed'
            job['error_message'] = None
            print(f"✅ Job {job['job_id']} completed: {message}")
        else:
            job['status'] = 'failed'
            job['error_message'] = message
            print(f"❌ Job {job['job_id']} failed: {message}")
        
        return job
    
    def process_batch(self, max_workers: Optional[int] = None) -> Dict:
        """Process all jobs in the queue using parallel processing."""
        if not self.job_queue:
            print("❌ No jobs in queue to process")
            return self.get_processing_stats()
        
        max_workers = max_workers or self.config['batch_settings']['max_workers']
        
        print(f"🚀 Starting batch processing with {max_workers} workers")
        print(f"📋 Processing {len(self.job_queue)} jobs")
        
        self.processing_stats['start_time'] = datetime.now()
        self.processing_stats['total_jobs'] = len(self.job_queue)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.process_single_job, job): job 
                for job in self.job_queue
            }
            
            # Process completed jobs
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    completed_job = future.result()
                    
                    if completed_job['status'] == 'completed':
                        self.completed_jobs.append(completed_job)
                        self.processing_stats['completed'] += 1
                    else:
                        # Retry failed jobs if configured
                        if (self.config['batch_settings']['retry_failed'] and 
                            completed_job['attempts'] < self.config['batch_settings']['max_retries']):
                            print(f"🔄 Retrying job {completed_job['job_id']}")
                            self.job_queue.append(completed_job)
                        else:
                            self.failed_jobs.append(completed_job)
                            self.processing_stats['failed'] += 1
                    
                except Exception as e:
                    print(f"❌ Job processing error: {e}")
                    self.failed_jobs.append(job)
                    self.processing_stats['failed'] += 1
        
        # Clear the processed jobs from queue
        self.job_queue = [job for job in self.job_queue if job['attempts'] < self.config['batch_settings']['max_retries']]
        
        self.processing_stats['end_time'] = datetime.now()
        self.processing_stats['processing_time'] = (
            self.processing_stats['end_time'] - self.processing_stats['start_time']
        ).total_seconds()
        
        # Generate outputs
        if self.config['output_settings']['generate_manifest']:
            self.generate_manifest()
        
        if self.config['output_settings']['generate_report']:
            self.generate_processing_report()
        
        print(f"\n🎉 Batch processing completed!")
        print(f"✅ Completed: {self.processing_stats['completed']}")
        print(f"❌ Failed: {self.processing_stats['failed']}")
        print(f"⏱️  Total time: {self.processing_stats['processing_time']:.1f}s")
        
        return self.get_processing_stats()
    
    def get_processing_stats(self) -> Dict:
        """Get current processing statistics."""
        return {
            'total_jobs': self.processing_stats['total_jobs'],
            'completed': self.processing_stats['completed'],
            'failed': self.processing_stats['failed'],
            'in_queue': len(self.job_queue),
            'processing_time': self.processing_stats['processing_time'],
            'success_rate': (self.processing_stats['completed'] / 
                           max(self.processing_stats['total_jobs'], 1)) * 100,
            'average_time_per_job': (self.processing_stats['processing_time'] / 
                                   max(self.processing_stats['completed'], 1)) if self.processing_stats['completed'] > 0 else 0
        }
    
    def generate_manifest(self) -> str:
        """Generate manifest files for completed jobs."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON manifest
        json_manifest = self.output_dir / "manifests" / f"batch_manifest_{timestamp}.json"
        manifest_data = {
            'generated_at': datetime.now().isoformat(),
            'batch_info': self.get_processing_stats(),
            'completed_jobs': self.completed_jobs,
            'failed_jobs': self.failed_jobs,
            'configuration': self.config
        }
        
        with open(json_manifest, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        # CSV manifest for easy viewing
        csv_manifest = self.output_dir / "manifests" / f"batch_manifest_{timestamp}.csv"
        
        csv_lines = ["job_id,text,output_file,status,processing_time,error_message"]
        
        for job in self.completed_jobs + self.failed_jobs:
            processing_time = ""
            if 'started_at' in job and 'completed_at' in job:
                start = datetime.fromisoformat(job['started_at'])
                end = datetime.fromisoformat(job['completed_at'])
                processing_time = str((end - start).total_seconds())
            
            csv_lines.append(
                f"{job['job_id']},\"{job['text'][:100]}\",{job['output_name']},"
                f"{job['status']},{processing_time},\"{job.get('error_message', '')}\""
            )
        
        with open(csv_manifest, 'w') as f:
            f.write('\n'.join(csv_lines))
        
        print(f"📋 Manifests generated:")
        print(f"   📄 JSON: {json_manifest}")
        print(f"   📊 CSV: {csv_manifest}")
        
        return str(json_manifest)
    
    def generate_processing_report(self) -> str:
        """Generate a comprehensive processing report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / "logs" / f"processing_report_{timestamp}.txt"
        
        stats = self.get_processing_stats()
        
        report = f"""
🚀 BATCH VOICE PROCESSING REPORT
{'='*50}

📊 PROCESSING STATISTICS
Total Jobs: {stats['total_jobs']}
Completed: {stats['completed']}
Failed: {stats['failed']}
Success Rate: {stats['success_rate']:.1f}%

⏱️ TIMING METRICS
Total Processing Time: {stats['processing_time']:.1f} seconds
Average Time per Job: {stats['average_time_per_job']:.1f} seconds
Jobs per Minute: {(stats['completed'] / max(stats['processing_time']/60, 1)):.1f}

📁 OUTPUT FILES
Audio Files: {self.output_dir}/audio/
Manifests: {self.output_dir}/manifests/
Logs: {self.output_dir}/logs/

✅ COMPLETED JOBS ({len(self.completed_jobs)})
"""
        
        for job in self.completed_jobs[:10]:  # Show first 10
            report += f"  {job['job_id']}: {job['text'][:50]}{'...' if len(job['text']) > 50 else ''}\n"
        
        if len(self.completed_jobs) > 10:
            report += f"  ... and {len(self.completed_jobs) - 10} more\n"
        
        if self.failed_jobs:
            report += f"\n❌ FAILED JOBS ({len(self.failed_jobs)})\n"
            for job in self.failed_jobs:
                report += f"  {job['job_id']}: {job.get('error_message', 'Unknown error')}\n"
        
        report += f"\n⚙️ CONFIGURATION USED\n"
        report += f"Max Workers: {self.config['batch_settings']['max_workers']}\n"
        report += f"Audio Format: {self.config['output_settings']['audio_format']}\n"
        report += f"Sample Rate: {self.config['output_settings']['sample_rate']}\n"
        
        report += f"\nReport generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📋 Processing report saved: {report_file}")
        return str(report_file)
    
    def clear_queue(self):
        """Clear all jobs from the queue."""
        cleared_count = len(self.job_queue)
        self.job_queue.clear()
        print(f"🗑️  Cleared {cleared_count} jobs from queue")
    
    def show_queue_status(self):
        """Display current queue status."""
        print(f"\n📋 Queue Status:")
        print(f"   Jobs in queue: {len(self.job_queue)}")
        print(f"   Completed jobs: {len(self.completed_jobs)}")
        print(f"   Failed jobs: {len(self.failed_jobs)}")
        
        if self.job_queue:
            print(f"\n🔜 Next jobs in queue:")
            for i, job in enumerate(self.job_queue[:5]):
                print(f"   {i+1}. {job['job_id']}: {job['text'][:50]}{'...' if len(job['text']) > 50 else ''}")
            if len(self.job_queue) > 5:
                print(f"   ... and {len(self.job_queue) - 5} more")
    
    def create_sample_config(self, config_file: str = "batch_config_sample.json"):
        """Create a sample configuration file."""
        sample_config = {
            "batch_settings": {
                "max_workers": 2,
                "timeout_per_job": 300,
                "retry_failed": True,
                "max_retries": 2,
                "save_intermediate": True
            },
            "model_settings": {
                "model_path": "output_fixed/run/training/GPT_XTTS_FT-*/best_model.pth",
                "config_path": "output_fixed/run/training/GPT_XTTS_FT-*/config.json",
                "language": "en",
                "speaker_wav": "data/dieck/dataset/dieck_001.wav"
            },
            "output_settings": {
                "audio_format": "wav",
                "sample_rate": 22050,
                "generate_manifest": True,
                "generate_report": True
            },
            "jobs": [
                {
                    "job_id": "demo_1",
                    "text": "Hello, this is a test of the batch voice processing system.",
                    "reference_audio": "data/dieck/dataset/dieck_001.wav",
                    "output_name": "demo_hello.wav"
                },
                {
                    "job_id": "demo_2",
                    "text": "This system can process multiple voice generation tasks efficiently.",
                    "output_name": "demo_efficient.wav"
                },
                {
                    "job_id": "demo_3",
                    "text": "Quality monitoring and batch processing make voice cloning scalable.",
                    "output_name": "demo_scalable.wav"
                }
            ]
        }
        
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        print(f"📝 Sample configuration created: {config_file}")
        return config_file

def main():
    """Main function for interactive batch processing."""
    print("🚀 Batch Voice Processor - Interactive Mode")
    print("=" * 50)
    
    processor = BatchVoiceProcessor()
    
    while True:
        print("\n📋 Available Commands:")
        print("1. Add single job")
        print("2. Add jobs from file")
        print("3. Create sample config")
        print("4. Load config file")
        print("5. Show queue status")
        print("6. Process batch")
        print("7. Clear queue")
        print("8. Generate manifest")
        print("9. Exit")
        
        choice = input("\n🎯 Enter your choice (1-9): ").strip()
        
        if choice == '1':
            text = input("📝 Enter text to synthesize: ").strip()
            ref_audio = input("📁 Reference audio path (optional): ").strip()
            
            if text:
                job_id = processor.add_job(
                    text=text,
                    reference_audio=ref_audio if ref_audio else None
                )
                print(f"✅ Job {job_id} added to queue")
            else:
                print("❌ Text cannot be empty")
        
        elif choice == '2':
            jobs_file = input("📁 Jobs file path (txt or json): ").strip()
            if os.path.exists(jobs_file):
                count = processor.load_jobs_from_file(jobs_file)
                print(f"✅ Loaded {count} jobs from {jobs_file}")
            else:
                print("❌ File not found")
        
        elif choice == '3':
            config_file = input("📁 Config file name (default: batch_config_sample.json): ").strip()
            if not config_file:
                config_file = "batch_config_sample.json"
            processor.create_sample_config(config_file)
        
        elif choice == '4':
            config_file = input("📁 Config file path: ").strip()
            if os.path.exists(config_file):
                processor.config = processor.load_config(config_file)
                # Load jobs from config
                for job_data in processor.config.get('jobs', []):
                    processor.add_job(
                        text=job_data['text'],
                        reference_audio=job_data.get('reference_audio'),
                        output_name=job_data.get('output_name'),
                        job_id=job_data.get('job_id')
                    )
            else:
                print("❌ Config file not found")
        
        elif choice == '5':
            processor.show_queue_status()
        
        elif choice == '6':
            if processor.job_queue:
                max_workers = input(f"🔧 Max workers (current: {processor.config['batch_settings']['max_workers']}): ").strip()
                if max_workers.isdigit():
                    max_workers = int(max_workers)
                else:
                    max_workers = None
                
                stats = processor.process_batch(max_workers)
                print(f"\n📊 Processing Summary:")
                print(f"   Success Rate: {stats['success_rate']:.1f}%")
                print(f"   Avg Time/Job: {stats['average_time_per_job']:.1f}s")
            else:
                print("❌ No jobs in queue")
        
        elif choice == '7':
            processor.clear_queue()
        
        elif choice == '8':
            if processor.completed_jobs or processor.failed_jobs:
                manifest_file = processor.generate_manifest()
                print(f"✅ Manifest generated: {manifest_file}")
            else:
                print("❌ No processed jobs to create manifest")
        
        elif choice == '9':
            print("👋 Batch processing session ended")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
