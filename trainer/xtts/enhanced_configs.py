from dataclasses import asdict, dataclass
from typing import Optional, Dict, List, Any

from coqpit import Coqpit, check_argument
from trainer import TrainerConfig


@dataclass
class EnhancedAudioConfig(Coqpit):
    """Enhanced audio configuration to fix robotic/echo issues and improve quality."""
    
    # Core STFT parameters - optimized for better quality
    fft_size: int = 1024
    win_length: int = 1024
    hop_length: int = 256  # Reduced for better temporal resolution
    frame_shift_ms: int = None
    frame_length_ms: int = None
    stft_pad_mode: str = "reflect"
    
    # Audio processing parameters - enhanced for better quality
    sample_rate: int = 22050
    resample: bool = False
    preemphasis: float = 0.0  # Disabled to reduce artifacts
    ref_level_db: int = 20
    do_sound_norm: bool = True  # Enabled for consistency
    log_func: str = "np.log10"
    
    # Enhanced silence trimming - more aggressive to remove artifacts
    do_trim_silence: bool = True
    trim_db: int = 35  # More aggressive trimming (was 45)
    
    # RMS volume normalization - enabled for consistency
    do_rms_norm: bool = True
    db_level: float = -20.0  # Standardized level
    
    # Griffin-Lim parameters - optimized to reduce artifacts
    power: float = 1.2  # Reduced from 1.5 to minimize artifacts
    griffin_lim_iters: int = 100  # Increased for better reconstruction
    
    # Mel-spectrogram parameters - enhanced resolution
    num_mels: int = 80
    mel_fmin: float = 50.0  # Better low-frequency handling
    mel_fmax: float = 8000.0  # Explicit max frequency
    spec_gain: int = 20
    do_amp_to_db_linear: bool = True
    do_amp_to_db_mel: bool = True
    
    # F0 parameters - improved pitch handling
    pitch_fmax: float = 800.0  # More realistic for most voices
    pitch_fmin: float = 50.0   # Better low pitch handling
    
    # Normalization parameters - enhanced for stability
    signal_norm: bool = True
    min_level_db: int = -100
    symmetric_norm: bool = True
    max_norm: float = 4.0
    clip_norm: bool = True
    stats_path: str = None

    def check_values(self):
        """Validate configuration parameters."""
        c = asdict(self)
        check_argument("num_mels", c, restricted=True, min_val=10, max_val=2056)
        check_argument("fft_size", c, restricted=True, min_val=128, max_val=4058)
        check_argument("sample_rate", c, restricted=True, min_val=512, max_val=100000)
        check_argument("preemphasis", c, restricted=True, min_val=0, max_val=1)
        check_argument("min_level_db", c, restricted=True, min_val=-1000, max_val=10)
        check_argument("ref_level_db", c, restricted=True, min_val=0, max_val=1000)
        check_argument("power", c, restricted=True, min_val=1, max_val=5)
        check_argument("griffin_lim_iters", c, restricted=True, min_val=10, max_val=1000)


@dataclass
class EnhancedDatasetConfig(Coqpit):
    """Enhanced dataset configuration for better language handling."""
    
    formatter: str = "coqui"
    dataset_name: str = "enhanced_ft_dataset"
    path: str = ""
    meta_file_train: str = ""
    meta_file_val: str = ""
    meta_file_attn_mask: str = ""
    language: str = "es"
    phonemizer: str = ""
    ignored_speakers: Optional[List[str]] = None
    
    # Enhanced language settings
    language_filter_threshold: float = 0.95  # Strict language filtering
    enable_language_conditioning: bool = True
    language_embedding_dim: int = 256


@dataclass
class EnhancedGPTArgs(Coqpit):
    """Enhanced GPT arguments to fix audio cutoff and language mixing."""
    
    # Audio length parameters - increased to prevent cutoff
    max_wav_length: int = 660000  # 30 seconds at 22050 Hz
    max_conditioning_length: int = 220000  # 10 seconds for conditioning
    min_conditioning_length: int = 66150   # 3 seconds minimum
    
    # Text parameters - enhanced for better language handling
    max_text_length: int = 350  # Increased from 200
    gpt_max_text_tokens: int = 402
    gpt_max_audio_tokens: int = 800  # Increased from 605
    gpt_max_prompt_tokens: int = 70
    
    # Model architecture
    gpt_layers: int = 30
    gpt_n_model_channels: int = 1024
    gpt_n_heads: int = 16
    
    # Token configuration
    gpt_number_text_tokens: int = 6681
    gpt_start_text_token: int = 261
    gpt_stop_text_token: int = 0
    gpt_num_audio_tokens: int = 1026
    gpt_start_audio_token: int = 1024
    gpt_stop_audio_token: int = 1025
    
    # Enhanced training parameters
    gpt_use_masking_gt_prompt_approach: bool = True
    gpt_use_perceiver_resampler: bool = True
    gpt_code_stride_len: int = 1024
    
    # Loss weights - balanced for better quality
    gpt_loss_text_ce_weight: float = 0.02  # Slightly increased
    gpt_loss_mel_ce_weight: float = 1.0
    
    # Debugging and file paths
    debug_loading_failures: bool = True
    mel_norm_file: str = ""
    dvae_checkpoint: str = ""
    xtts_checkpoint: str = ""
    tokenizer_file: str = ""
    vocoder: str = ""
    
    # Audio processing
    input_sample_rate: int = 22050
    output_sample_rate: int = 24000
    output_hop_length: int = 256
    decoder_input_dim: int = 1024
    d_vector_dim: int = 512
    cond_d_vector_in_each_upsampling_layer: bool = True
    duration_const: int = 102400


@dataclass 
class EnhancedXTTSConfig(TrainerConfig):
    """Enhanced XTTS configuration with optimized hyperparameters to fix common issues."""
    
    # Core training parameters
    epochs: int = 20
    batch_size: int = 6  # Slightly increased
    eval_batch_size: int = 3
    grad_clip: float = 1.0  # Increased for stability
    
    # Enhanced optimizer settings
    optimizer: str = "AdamW"
    lr: float = 1.5e-06  # Slightly reduced for stability
    optimizer_params: Dict[str, Any] = None
    
    # Learning rate scheduling
    lr_scheduler: str = "CosineAnnealingLR"
    lr_scheduler_params: Dict[str, Any] = None
    scheduler_after_epoch: bool = True
    
    # Enhanced precision settings
    mixed_precision: bool = False
    precision: str = "fp16"
    use_grad_scaler: bool = True  # Enable gradient scaling
    
    # Audio and model settings
    model: str = "xtts"
    audio: EnhancedAudioConfig = None
    model_args: EnhancedGPTArgs = None
    
    # Enhanced inference parameters to fix language mixing
    temperature: float = 0.75  # Reduced for more consistent output
    length_penalty: float = 1.2  # Increased to encourage completion
    repetition_penalty: float = 2.5  # Increased to reduce repetition
    top_k: int = 40  # Reduced for more focused sampling
    top_p: float = 0.8  # Reduced for more consistent output
    num_gpt_outputs: int = 1
    
    # Enhanced conditioning parameters
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 12  # Increased for better reference
    sound_norm_refs: bool = True  # Enable reference normalization
    
    # Enhanced training settings
    use_noise_augment: bool = True
    num_loader_workers: int = 4  # Increased for better data loading
    training_seed: int = 1337  # Fixed seed for reproducibility
    
    # Logging and saving
    print_step: int = 25
    plot_step: int = 100
    save_step: int = 250  # More frequent saves
    save_n_checkpoints: int = 15  # Keep more checkpoints
    
    def __post_init__(self):
        if self.audio is None:
            self.audio = EnhancedAudioConfig()
        if self.model_args is None:
            self.model_args = EnhancedGPTArgs()
        if self.optimizer_params is None:
            self.optimizer_params = {
                "betas": [0.9, 0.999],
                "weight_decay": 0.01,
                "eps": 1e-8
            }
        if self.lr_scheduler_params is None:
            self.lr_scheduler_params = {
                "T_max": 2000,
                "eta_min": 1e-7
            }
