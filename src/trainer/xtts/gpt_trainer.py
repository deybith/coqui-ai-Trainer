import gc
import logging
import os

from trainer import Trainer, TrainerArgs

from .xtts.shared_configs import BaseDatasetConfig
from .xtts.datasets import load_tts_samples
from .xtts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from .xtts.models.xtts import XttsAudioConfig
from TTS.utils.manage import ModelManager

logger = logging.getLogger("trainer")


def train_gpt(language, num_epochs, batch_size, grad_acumm, train_csv, eval_csv, output_path, max_audio_length=255995, training_seed=54321, config=None):
    """
    Enhanced GPT training function with support for optimized configurations.
    
    Args:
        config: Optional EnhancedXTTSConfig object with optimized parameters
    """
    #  Logging parameters
    RUN_NAME = "GPT_XTTS_FT"
    PROJECT_NAME = "XTTS_trainer"
    DASHBOARD_LOGGER = "tensorboard"
    LOGGER_URI = None

    # Set here the path that the checkpoints will be saved. Default: ./run/training/
    OUT_PATH = os.path.join(output_path, "run", "training")

    # Training Parameters
    OPTIMIZER_WD_ONLY_ON_WEIGHTS = False  # for multi-gpu training
    START_WITH_EVAL = False  # if True it will start with evaluation
    BATCH_SIZE = batch_size  # set here the batch size
    GRAD_ACUMM_STEPS = grad_acumm  # set here the grad accumulation steps

    # Define here the dataset that you want to use for the fine-tuning on.
    config_dataset = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="ft_dataset",
        path=os.path.dirname(train_csv),
        meta_file_train=os.path.basename(train_csv),
        meta_file_val=os.path.basename(eval_csv),
        language=language,
    )

    # Add here the configs of the datasets
    DATASETS_CONFIG_LIST = [config_dataset]

    # Define the path where XTTS v2.0.1 files will be downloaded
    CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files/")
    os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

    # DVAE files
    DVAE_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/dvae.pth"
    MEL_NORM_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/mel_stats.pth"

    # Set the path to the downloaded files
    DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
    MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

    # download DVAE files if needed
    if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
        logger.info(" > Downloading DVAE files!")
        ModelManager._download_model_files(
            [MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # Download XTTS v2.0 checkpoint if needed
    TOKENIZER_FILE_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/vocab.json"
    XTTS_CHECKPOINT_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/model.pth"
    XTTS_CONFIG_LINK = "https://huggingface.co/coqui/XTTS-v2/resolve/main/config.json"

    # XTTS transfer learning parameters: You we need to provide the paths of XTTS model checkpoint that you want to do the fine tuning.
    TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))  # vocab.json file
    XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))  # model.pth file
    XTTS_CONFIG_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CONFIG_LINK))  # config.json file

    # download XTTS v2.0 files if needed
    if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
        logger.info(" > Downloading XTTS v2.0 files!")
        ModelManager._download_model_files(
            [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK, XTTS_CONFIG_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
        )

    # init args and config - use enhanced config if provided
    if config and hasattr(config, 'gpt'):
        # Use enhanced GPT configuration
        enhanced_gpt = config.gpt
        logger.info("🚀 Using enhanced GPT configuration")
        
        model_args = GPTArgs(
            max_conditioning_length=enhanced_gpt.max_conditioning_length,
            min_conditioning_length=enhanced_gpt.min_conditioning_length,
            debug_loading_failures=False,
            max_wav_length=enhanced_gpt.max_wav_length,
            max_text_length=enhanced_gpt.max_text_length,
            mel_norm_file=MEL_NORM_FILE,
            dvae_checkpoint=DVAE_CHECKPOINT,
            xtts_checkpoint=XTTS_CHECKPOINT,
            tokenizer_file=TOKENIZER_FILE,
            gpt_num_audio_tokens=enhanced_gpt.gpt_num_audio_tokens,
            gpt_start_audio_token=enhanced_gpt.gpt_start_audio_token,
            gpt_stop_audio_token=enhanced_gpt.gpt_stop_audio_token,
            gpt_use_masking_gt_prompt_approach=enhanced_gpt.gpt_use_masking_gt_prompt_approach,
            gpt_use_perceiver_resampler=enhanced_gpt.gpt_use_perceiver_resampler,
            gpt_max_audio_tokens=enhanced_gpt.gpt_max_audio_tokens,
            gpt_max_text_tokens=enhanced_gpt.gpt_max_text_tokens,
            temperature=enhanced_gpt.temperature,
            repetition_penalty=enhanced_gpt.repetition_penalty,
            top_k=enhanced_gpt.top_k,
            top_p=enhanced_gpt.top_p,
        )
        
        # Use enhanced audio config if available
        if hasattr(config, 'audio'):
            enhanced_audio = config.audio
            logger.info("🎵 Using enhanced audio configuration")
            audio_config = XttsAudioConfig(
                sample_rate=enhanced_audio.sample_rate,
                dvae_sample_rate=enhanced_audio.sample_rate,
                output_sample_rate=enhanced_audio.output_sample_rate,
                hop_length=enhanced_audio.hop_length,
                win_length=enhanced_audio.win_length,
                fft_size=enhanced_audio.fft_size,
                mel_fmin=enhanced_audio.mel_fmin,
                mel_fmax=enhanced_audio.mel_fmax,
                num_mels=enhanced_audio.num_mels,
                do_sound_norm=enhanced_audio.do_sound_norm,
                do_rms_norm=enhanced_audio.do_rms_norm,
                do_dynamic_range_compression=enhanced_audio.do_dynamic_range_compression,
                trim_db=enhanced_audio.trim_db,
                power=enhanced_audio.power,
                griffin_lim_iters=enhanced_audio.griffin_lim_iters,
            )
        else:
            audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
            
    else:
        # Use original configuration
        logger.info("📝 Using original GPT configuration")
        model_args = GPTArgs(
            max_conditioning_length=132300,  # 6 secs
            min_conditioning_length=66150,  # 3 secs
            debug_loading_failures=False,
            max_wav_length=max_audio_length,  # ~11.6 seconds
            max_text_length=200,
            mel_norm_file=MEL_NORM_FILE,
            dvae_checkpoint=DVAE_CHECKPOINT,
            xtts_checkpoint=XTTS_CHECKPOINT,  # checkpoint path of the model that you want to fine-tune
            tokenizer_file=TOKENIZER_FILE,
            gpt_num_audio_tokens=1026,
            gpt_start_audio_token=1024,
            gpt_stop_audio_token=1025,
            gpt_use_masking_gt_prompt_approach=True,
            gpt_use_perceiver_resampler=True,
        )
        # define audio config
        audio_config = XttsAudioConfig(sample_rate=22050, dvae_sample_rate=22050, output_sample_rate=24000)
    # training parameters config
    config = GPTTrainerConfig(
        epochs=num_epochs,
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name=PROJECT_NAME,
        run_description="GPT XTTS training",
        dashboard_logger=DASHBOARD_LOGGER,
        logger_uri=LOGGER_URI,
        audio=audio_config,
        
        # Training settings
        batch_size=BATCH_SIZE,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        training_seed=training_seed,
        
        # Eval settings
        eval_split_max_size=256,
        print_eval=False,
        
        # Logging
        print_step=50,
        plot_step=100,
        save_step=1000,
        
        # Optimizer settings
        optimizer="AdamW",
        optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=5e-06,  # learning rate
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
        
        test_sentences=[]
    )

    # init the model from config
    model = GPTTrainer.init_from_config(config)

    # load training samples
    train_samples, eval_samples = load_tts_samples(
        DATASETS_CONFIG_LIST,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,  # xtts checkpoint is restored via xtts_checkpoint key so no need of restore it using Trainer restore_path parameter
            skip_train_epoch=False,
            start_with_eval=START_WITH_EVAL,
            grad_accum_steps=GRAD_ACUMM_STEPS,
        ),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()

    # get the longest text audio file to use as speaker reference
    samples_len = [len(item["text"].split(" ")) for item in train_samples]
    longest_text_idx = samples_len.index(max(samples_len))
    speaker_ref = train_samples[longest_text_idx]["audio_file"]

    trainer_out_path = trainer.output_path

    # deallocate VRAM and RAM
    del model, trainer, train_samples, eval_samples
    gc.collect()

    return XTTS_CONFIG_FILE, XTTS_CHECKPOINT, TOKENIZER_FILE, trainer_out_path, speaker_ref
