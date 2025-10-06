"""
Fine-tuning module for Van Gogh Image Generator
Implements LoRA and QLoRA techniques
"""
import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from diffusers import (
    StableDiffusionPipeline,
    DDPMScheduler,
    UNet2DConditionModel,
    AutoencoderKL
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
from accelerate import Accelerator
from datasets import Dataset as HFDataset
import wandb
from tqdm import tqdm

from config import training_config, env_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VanGoghDataset(Dataset):
    """Custom dataset for Van Gogh style images"""
    
    def __init__(self, data_dir: str, tokenizer, max_length: int = 77):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load metadata
        with open(self.data_dir / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        self.images = self.metadata['images']
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_info = self.images[idx]
        
        # Tokenize prompt
        prompt = image_info['prompt']
        tokens = self.tokenizer(
            prompt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(),
            "attention_mask": tokens["attention_mask"].squeeze(),
            "prompt": prompt,
            "image_path": image_info['filepath']
        }

class VanGoghFineTuner:
    """Handles fine-tuning of models with Van Gogh style"""
    
    def __init__(self, config: training_config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.accelerator = Accelerator()
        
        # Initialize wandb
        if env_config.get("wandb_entity"):
            wandb.init(
                project=env_config["wandb_project"],
                entity=env_config["wandb_entity"],
                config=vars(config)
            )
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Fine-tuning configuration: {vars(config)}")
    
    def setup_model_and_tokenizer(self):
        """Setup the base model and tokenizer"""
        logger.info(f"Loading base model: {self.config.base_model}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            use_fast=False
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        if self.config.use_qlora:
            # Load model in 4-bit for QLoRA
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                device_map="auto",
                quantization_config={
                    "load_in_4bit": True,
                    "bnb_4bit_compute_dtype": torch.float16,
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_quant_type": "nf4"
                }
            )
        else:
            # Load model in full precision for LoRA
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def setup_lora_config(self):
        """Setup LoRA configuration"""
        logger.info("Setting up LoRA configuration...")
        
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Apply LoRA to model
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        logger.info("LoRA configuration applied successfully")
    
    def prepare_dataset(self, data_dir: str) -> VanGoghDataset:
        """Prepare the dataset for training"""
        logger.info(f"Preparing dataset from: {data_dir}")
        
        dataset = VanGoghDataset(data_dir, self.tokenizer)
        logger.info(f"Dataset prepared with {len(dataset)} samples")
        
        return dataset
    
    def setup_training_args(self) -> TrainingArguments:
        """Setup training arguments"""
        output_dir = Path(self.config.output_dir) / self.config.model_name
        
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=100,
            weight_decay=0.01,
            fp16=self.config.mixed_precision == "fp16",
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="wandb" if env_config.get("wandb_entity") else None,
        )
        
        return training_args
    
    def train(self, data_dir: str):
        """Main training function"""
        logger.info("Starting fine-tuning process...")
        
        try:
            # Setup model and tokenizer
            self.setup_model_and_tokenizer()
            
            # Setup LoRA
            self.setup_lora_config()
            
            # Prepare dataset
            dataset = self.prepare_dataset(data_dir)
            
            # Setup training arguments
            training_args = self.setup_training_args()
            
            # Setup trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                eval_dataset=dataset,  # Using same dataset for eval (in production, use separate validation set)
                tokenizer=self.tokenizer,
                data_collator=DataCollatorForLanguageModeling(
                    tokenizer=self.tokenizer,
                    mlm=False
                ),
            )
            
            # Start training
            logger.info("Starting training...")
            trainer.train()
            
            # Save the fine-tuned model
            output_dir = Path(self.config.output_dir) / self.config.model_name
            trainer.save_model(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))
            
            # Save LoRA weights separately
            lora_output_dir = output_dir / "lora_weights"
            self.model.save_pretrained(str(lora_output_dir))
            
            logger.info(f"Training completed! Model saved to: {output_dir}")
            
            # Log final metrics
            final_metrics = trainer.evaluate()
            logger.info(f"Final evaluation metrics: {final_metrics}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def save_training_config(self, output_dir: str):
        """Save training configuration for later use"""
        config_path = Path(output_dir) / "training_config.json"
        
        config_dict = {
            "base_model": self.config.base_model,
            "model_name": self.config.model_name,
            "lora_config": {
                "r": self.config.lora_r,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout
            },
            "training_params": {
                "learning_rate": self.config.learning_rate,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size
            },
            "use_qlora": self.config.use_qlora
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Training configuration saved to: {config_path}")

def main():
    """Main function for fine-tuning"""
    # Check if dataset exists
    data_dir = "data"
    if not Path(data_dir).exists():
        print("Dataset not found. Please run data_preparation.py first.")
        return
    
    # Initialize fine-tuner
    fine_tuner = VanGoghFineTuner(training_config)
    
    # Start training
    if fine_tuner.train(data_dir):
        print("Fine-tuning completed successfully!")
        
        # Save configuration
        output_dir = Path(training_config.output_dir) / training_config.model_name
        fine_tuner.save_training_config(str(output_dir))
    else:
        print("Fine-tuning failed!")

if __name__ == "__main__":
    main()
