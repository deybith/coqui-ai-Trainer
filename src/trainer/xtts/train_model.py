import logging
import os
import traceback

import torch

from .xtts.gpt_trainer import train_gpt

logger = logging.getLogger("trainer")

def clear_gpu_cache():
    # clear the GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_xtts(config):
    """
    Enhanced XTTS training function using optimized configurations.
    
    Args:
        config: EnhancedXTTSConfig object with optimized parameters
    
    Returns:
        Trained model paths and configuration
    """
    clear_gpu_cache()
    
    try:
        logger.info("🚀 Starting enhanced XTTS training...")
        
        # Extract dataset configuration
        if not config.datasets or len(config.datasets) == 0:
            raise ValueError("No datasets configured")
        
        dataset = config.datasets[0]
        train_csv = dataset["meta_file_train"]
        eval_csv = dataset["meta_file_val"]
        language = dataset["language"]
        
        if not train_csv or not eval_csv:
            raise ValueError("Train CSV and Eval CSV must be specified")
        
        # Use enhanced GPT configuration
        gpt_config = config.model_args if hasattr(config, 'model_args') and config.model_args else config
        
        logger.info("📊 Training with enhanced parameters:")
        logger.info("  • Max audio length: %.1fs", gpt_config.max_wav_length / 22050)
        logger.info("  • Max audio tokens: %d", gpt_config.gpt_max_audio_tokens)
        logger.info("  • Repetition penalty: %.2f", config.repetition_penalty)
        logger.info("  • Temperature: %.2f", config.temperature)
        logger.info("  • Batch size: %d", config.batch_size)
        logger.info("  • Learning rate: %f", config.lr)
        
        # Train with enhanced configuration
        config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
            language=language,
            num_epochs=config.epochs,
            batch_size=config.batch_size,
            grad_acumm=getattr(config, 'grad_acumm', 1),
            train_csv=train_csv,
            eval_csv=eval_csv,
            output_path=config.output_path,
            max_audio_length=gpt_config.max_wav_length,
            config=config  # Pass the full enhanced config
        )
        
        # Copy configuration files
        os.system(f"cp {config_path} {exp_path}")
        os.system(f"cp {vocab_file} {exp_path}")
        
        ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
        
        logger.info("✅ Enhanced model training completed successfully!")
        clear_gpu_cache()
        
        return {
            "status": "success",
            "message": "Enhanced model training done!",
            "config_path": config_path,
            "vocab_file": vocab_file,
            "checkpoint": ft_xtts_checkpoint,
            "speaker_wav": speaker_wav,
            "exp_path": exp_path
        }
        
    except Exception as e:
        logger.exception("Exception occurred during enhanced training:")
        error = traceback.format_exc()
        clear_gpu_cache()
        
        return {
            "status": "error",
            "message": f"Enhanced training failed: {str(e)}",
            "error": error
        }


def train_model(
                language, train_csv, eval_csv, num_epochs, batch_size, grad_acumm, output_path, max_audio_length
            ):
                clear_gpu_cache()
                if not train_csv or not eval_csv:
                    return (
                        "You need to run the data processing step or manually set `Train CSV` and `Eval CSV` fields !",
                        "",
                        "",
                        "",
                        "",
                    )
                try:
                    # convert seconds to waveform frames
                    max_audio_length = int(max_audio_length * 22050)
                    config_path, original_xtts_checkpoint, vocab_file, exp_path, speaker_wav = train_gpt(
                        language,
                        num_epochs,
                        batch_size,
                        grad_acumm,
                        train_csv,
                        eval_csv,
                        output_path=output_path,
                        max_audio_length=max_audio_length,
                    )
                except:
                    logger.exception("Training was interrupted due to an error:")
                    error = traceback.format_exc()
                    return (
                        f"The training was interrupted due an error !! Please check the console to check the full error message! \n Error summary: {error}",
                        "",
                        "",
                        "",
                        "",
                    )

                # copy original files to avoid parameters changes issues
                os.system(f"cp {config_path} {exp_path}")
                os.system(f"cp {vocab_file} {exp_path}")

                ft_xtts_checkpoint = os.path.join(exp_path, "best_model.pth")
                logger.info("Model training done!")
                clear_gpu_cache()
                return "Model training done!", config_path, vocab_file, ft_xtts_checkpoint, speaker_wav