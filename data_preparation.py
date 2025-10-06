"""
Data preparation module for Van Gogh Image Generator
Handles dataset creation, preprocessing, and validation
"""
import os
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
import logging

from config import training_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VanGoghDataPreparator:
    """Handles preparation of Van Gogh style images for fine-tuning"""
    
    def __init__(self, data_dir: str = "data", min_images: int = 30):
        self.data_dir = Path(data_dir)
        self.min_images = min_images
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.metadata_file = self.data_dir / "metadata.json"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Van Gogh image sources (public domain/CC licensed)
        self.image_sources = [
            "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ec/Moulin_de_la_Galette%2C_by_Vincent_van_Gogh%2C_from_Art_Institute_of_Chicago.jpg/512px-Moulin_de_la_Galette%2C_by_Vincent_van_Gogh%2C_from_Art_Institute_of_Chicago.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_Self-Portrait_-_Google_Art_Project.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/59/Vincent_van_Gogh_-_The_Starry_Night_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_The_Starry_Night_-_Google_Art_Project.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Vincent_van_Gogh_-_Sunflowers_-_VGM_F458.jpg/512px-Vincent_van_Gogh_-_Sunflowers_-_VGM_F458.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Vincent_van_Gogh_-_Wheatfield_with_Cypresses_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_Wheatfield_with_Cypresses_-_Google_Art_Project.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/Vincent_van_Gogh_-_The_Bedroom_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_The_Bedroom_-_Google_Art_Project.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Vincent_van_Gogh_-_Caf%C3%A9_Terrace_at_Night_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_Caf%C3%A9_Terrace_at_Night_-_Google_Art_Project.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/21/Vincent_van_Gogh_-_Almond_Blossom_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_Almond_Blossom_-_Google_Art_Project.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Vincent_van_Gogh_-_Irises_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_Irises_-_Google_Art_Project.jpg",
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Vincent_van_Gogh_-_The_Red_Vineyard_-_Google_Art_Project.jpg/512px-Vincent_van_Gogh_-_The_Red_Vineyard_-_Google_Art_Project.jpg"
        ]
        
    def download_images(self) -> int:
        """Download Van Gogh images from public sources"""
        logger.info(f"Downloading {len(self.image_sources)} Van Gogh images...")
        
        downloaded_count = 0
        for i, url in enumerate(tqdm(self.image_sources, desc="Downloading images")):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()
                
                # Save image
                image_path = self.raw_dir / f"vangogh_{i:03d}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                downloaded_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
                
        logger.info(f"Successfully downloaded {downloaded_count} images")
        return downloaded_count
    
    def preprocess_images(self) -> int:
        """Preprocess downloaded images for training"""
        logger.info("Preprocessing images for training...")
        
        processed_count = 0
        image_files = list(self.raw_dir.glob("*.jpg")) + list(self.raw_dir.glob("*.png"))
        
        for image_file in tqdm(image_files, desc="Preprocessing images"):
            try:
                # Load image
                image = Image.open(image_file)
                
                # Convert to RGB if necessary
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                # Resize to target size
                target_size = training_config.image_size
                image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
                
                # Save processed image
                output_path = self.processed_dir / f"{image_file.stem}_processed.png"
                image.save(output_path, "PNG", quality=95)
                
                processed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process {image_file}: {e}")
                
        logger.info(f"Successfully processed {processed_count} images")
        return processed_count
    
    def create_metadata(self) -> Dict:
        """Create metadata for the dataset"""
        logger.info("Creating dataset metadata...")
        
        processed_files = list(self.processed_dir.glob("*_processed.png"))
        
        metadata = {
            "dataset_name": "vangogh_style_dataset",
            "description": "Fine-tuned dataset for Van Gogh style image generation",
            "num_images": len(processed_files),
            "image_size": training_config.image_size,
            "style": "Vincent van Gogh post-impressionist",
            "images": []
        }
        
        for image_file in processed_files:
            image_info = {
                "filename": image_file.name,
                "filepath": str(image_file),
                "prompt": "a painting in the style of Vincent van Gogh",
                "negative_prompt": "low quality, blurry, distorted, ugly, bad anatomy",
                "style_tags": ["vangogh", "post-impressionist", "oil painting", "artistic"]
            }
            metadata["images"].append(image_info)
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Metadata saved with {len(metadata['images'])} images")
        return metadata
    
    def validate_dataset(self) -> bool:
        """Validate the prepared dataset"""
        logger.info("Validating dataset...")
        
        # Check minimum image count
        processed_files = list(self.processed_dir.glob("*_processed.png"))
        if len(processed_files) < self.min_images:
            logger.error(f"Dataset has {len(processed_files)} images, minimum required is {self.min_images}")
            return False
        
        # Check image quality
        valid_images = 0
        for image_file in processed_files:
            try:
                image = Image.open(image_file)
                if image.size == (training_config.image_size, training_config.image_size):
                    valid_images += 1
            except Exception as e:
                logger.warning(f"Invalid image {image_file}: {e}")
        
        if valid_images < self.min_images:
            logger.error(f"Only {valid_images} valid images found, minimum required is {self.min_images}")
            return False
        
        logger.info(f"Dataset validation passed: {valid_images} valid images")
        return True
    
    def prepare_dataset(self) -> bool:
        """Complete dataset preparation pipeline"""
        logger.info("Starting dataset preparation...")
        
        # Download images
        downloaded = self.download_images()
        if downloaded == 0:
            logger.error("No images downloaded")
            return False
        
        # Preprocess images
        processed = self.preprocess_images()
        if processed == 0:
            logger.error("No images processed")
            return False
        
        # Create metadata
        metadata = self.create_metadata()
        
        # Validate dataset
        if not self.validate_dataset():
            return False
        
        logger.info("Dataset preparation completed successfully!")
        logger.info(f"Final dataset: {metadata['num_images']} images")
        return True
    
    def get_dataset_info(self) -> Dict:
        """Get information about the prepared dataset"""
        if not self.metadata_file.exists():
            return {"error": "Dataset not prepared"}
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        return metadata

def main():
    """Main function for data preparation"""
    preparator = VanGoghDataPreparator()
    
    if preparator.prepare_dataset():
        info = preparator.get_dataset_info()
        print(f"Dataset prepared successfully: {info['num_images']} images")
        print(f"Dataset location: {preparator.data_dir}")
    else:
        print("Dataset preparation failed")

if __name__ == "__main__":
    main()
