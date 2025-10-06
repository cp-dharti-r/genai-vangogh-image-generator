"""
Configuration file for Van Gogh Image Generator
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """Configuration for fine-tuning"""
    # Model settings
    base_model: str = "runwayml/stable-diffusion-v1-5"
    model_name: str = "vangogh-style"
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    num_epochs: int = 100
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # LoRA settings
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # QLoRA settings
    use_qlora: bool = True
    bits: int = 4
    group_size: int = 128
    
    # Data settings
    min_images: int = 30
    image_size: int = 512
    validation_split: float = 0.1
    
    # Output settings
    output_dir: str = "outputs"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 10
    
    # Hardware settings
    mixed_precision: str = "fp16"
    use_xformers: bool = True
    gradient_checkpointing: bool = True

@dataclass
class GenerationConfig:
    """Configuration for image generation"""
    # Generation parameters
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    num_images_per_prompt: int = 1
    
    # Style settings
    style_prompt: str = "in the style of Vincent van Gogh, post-impressionist painting"
    
    # Output settings
    output_format: str = "png"
    output_quality: int = 95

@dataclass
class WebConfig:
    """Configuration for web interface"""
    host: str = "0.0.0.0"
    port: int = 7860
    debug: bool = False
    share: bool = False

# Environment variables
def get_env_config():
    """Get configuration from environment variables"""
    return {
        "base_model": os.getenv("BASE_MODEL", "runwayml/stable-diffusion-v1-5"),
        "output_dir": os.getenv("OUTPUT_DIR", "outputs"),
        "wandb_project": os.getenv("WANDB_PROJECT", "vangogh-generator"),
        "wandb_entity": os.getenv("WANDB_ENTITY", None),
    }

# Default configurations
training_config = TrainingConfig()
generation_config = GenerationConfig()
web_config = WebConfig()
env_config = get_env_config()
