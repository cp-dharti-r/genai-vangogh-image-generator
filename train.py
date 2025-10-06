#!/usr/bin/env python3
"""
Main training script for Van Gogh Image Generator
Orchestrates the entire fine-tuning pipeline
"""
import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data_preparation import VanGoghDataPreparator
from fine_tuning import VanGoghFineTuner
from config import training_config

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description="Van Gogh Image Generator Training")
    parser.add_argument(
        "--skip-data-prep", 
        action="store_true", 
        help="Skip data preparation if dataset already exists"
    )
    parser.add_argument(
        "--skip-training", 
        action="store_true", 
        help="Skip training if only data preparation is needed"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="data",
        help="Directory for dataset (default: data)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default=None,
        help="Output directory for trained model (default: from config)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs (default: from config)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=None,
        help="Learning rate (default: from config)"
    )
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.output_dir:
        training_config.output_dir = args.output_dir
    if args.epochs:
        training_config.num_epochs = args.epochs
    if args.learning_rate:
        training_config.learning_rate = args.learning_rate
    
    print("🎨 Van Gogh Image Generator - Training Pipeline")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {training_config.output_dir}")
    print(f"Training epochs: {training_config.num_epochs}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Using QLoRA: {training_config.use_qlora}")
    print("=" * 60)
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        print("\n📊 Step 1: Data Preparation")
        print("-" * 30)
        
        data_preparator = VanGoghDataPreparator(
            data_dir=args.data_dir,
            min_images=training_config.min_images
        )
        
        if data_preparator.prepare_dataset():
            dataset_info = data_preparator.get_dataset_info()
            print(f"✅ Dataset prepared successfully: {dataset_info['num_images']} images")
        else:
            print("❌ Dataset preparation failed!")
            return False
    else:
        print("\n📊 Step 1: Data Preparation (Skipped)")
        print("Checking if dataset exists...")
        
        data_dir = Path(args.data_dir)
        if not data_dir.exists() or not (data_dir / "metadata.json").exists():
            print("❌ Dataset not found! Please run without --skip-data-prep")
            return False
        
        print("✅ Dataset found, proceeding to training...")
    
    # Step 2: Fine-tuning
    if not args.skip_training:
        print("\n🤖 Step 2: Fine-tuning")
        print("-" * 30)
        
        fine_tuner = VanGoghFineTuner(training_config)
        
        if fine_tuner.train(args.data_dir):
            print("✅ Fine-tuning completed successfully!")
            
            # Save training configuration
            output_dir = Path(training_config.output_dir) / training_config.model_name
            fine_tuner.save_training_config(str(output_dir))
            
            print(f"🎯 Model saved to: {output_dir}")
            print(f"📁 LoRA weights saved to: {output_dir / 'lora_weights'}")
        else:
            print("❌ Fine-tuning failed!")
            return False
    else:
        print("\n🤖 Step 2: Fine-tuning (Skipped)")
    
    print("\n🎉 Training pipeline completed successfully!")
    print("\nNext steps:")
    print("1. Run 'python web_app.py' to launch the web interface")
    print("2. Or use 'python image_generator.py' for programmatic access")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        print(f"\n❌ Training pipeline failed: {e}")
        sys.exit(1)
