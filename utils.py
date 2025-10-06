"""
Utility functions for Van Gogh Image Generator
Helper functions for image processing, model evaluation, and common operations
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, structural_similarity
import cv2

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def enhance_image(image: Image.Image, enhancement_factor: float = 1.2) -> Image.Image:
        """Enhance image quality"""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(enhancement_factor)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(enhancement_factor)
        
        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(enhancement_factor)
        
        return image
    
    @staticmethod
    def apply_van_gogh_filter(image: Image.Image) -> Image.Image:
        """Apply a Van Gogh style filter to enhance artistic appearance"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Enhance saturation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)  # Increase saturation
        img_array = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Apply slight blur for painterly effect
        img_array = cv2.GaussianBlur(img_array, (3, 3), 0.5)
        
        # Enhance edges
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        img_array = cv2.filter2D(img_array, -1, kernel)
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def create_image_grid(images: List[Image.Image], cols: int = 2) -> Image.Image:
        """Create a grid of images"""
        if not images:
            return None
        
        # Calculate grid dimensions
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        # Get image dimensions
        img_width, img_height = images[0].size
        
        # Create grid image
        grid_width = cols * img_width
        grid_height = rows * img_height
        grid_image = Image.new('RGB', (grid_width, grid_height))
        
        # Place images in grid
        for i, img in enumerate(images):
            x = (i % cols) * img_width
            y = (i // cols) * img_height
            grid_image.paste(img, (x, y))
        
        return grid_image
    
    @staticmethod
    def resize_image(image: Image.Image, target_size: Tuple[int, int]) -> Image.Image:
        """Resize image to target size while maintaining aspect ratio"""
        # Calculate aspect ratio
        img_ratio = image.width / image.height
        target_ratio = target_size[0] / target_size[1]
        
        if img_ratio > target_ratio:
            # Image is wider than target
            new_width = target_size[0]
            new_height = int(target_size[0] / img_ratio)
        else:
            # Image is taller than target
            new_height = target_size[1]
            new_width = int(target_size[1] * img_ratio)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create target image with padding
        target_image = Image.new('RGB', target_size, (255, 255, 255))
        
        # Center the resized image
        x = (target_size[0] - new_width) // 2
        y = (target_size[1] - new_height) // 2
        target_image.paste(resized, (x, y))
        
        return target_image

class ModelEvaluator:
    """Utility class for model evaluation and metrics"""
    
    @staticmethod
    def calculate_image_similarity(img1: Image.Image, img2: Image.Image) -> float:
        """Calculate structural similarity between two images"""
        # Convert to numpy arrays
        img1_array = np.array(img1.convert('L'))  # Convert to grayscale
        img2_array = np.array(img2.convert('L'))
        
        # Ensure same size
        if img1_array.shape != img2_array.shape:
            img2_array = cv2.resize(img2_array, (img1_array.shape[1], img1_array.shape[0]))
        
        # Calculate SSIM
        ssim = structural_similarity(img1_array, img2_array)
        return ssim
    
    @staticmethod
    def calculate_mse(img1: Image.Image, img2: Image.Image) -> float:
        """Calculate Mean Squared Error between two images"""
        # Convert to numpy arrays
        img1_array = np.array(img1.convert('L'))
        img2_array = np.array(img2.convert('L'))
        
        # Ensure same size
        if img1_array.shape != img2_array.shape:
            img2_array = cv2.resize(img2_array, (img1_array.shape[1], img1_array.shape[0]))
        
        # Calculate MSE
        mse = mean_squared_error(img1_array, img2_array)
        return mse
    
    @staticmethod
    def evaluate_generation_quality(generated_images: List[Image.Image], 
                                  reference_images: List[Image.Image]) -> Dict[str, float]:
        """Evaluate the quality of generated images against reference images"""
        if len(generated_images) != len(reference_images):
            logger.warning("Number of generated and reference images don't match")
            return {}
        
        ssim_scores = []
        mse_scores = []
        
        for gen_img, ref_img in zip(generated_images, reference_images):
            ssim = ModelEvaluator.calculate_image_similarity(gen_img, ref_img)
            mse = ModelEvaluator.calculate_mse(gen_img, ref_img)
            
            ssim_scores.append(ssim)
            mse_scores.append(mse)
        
        # Calculate statistics
        metrics = {
            'mean_ssim': np.mean(ssim_scores),
            'std_ssim': np.std(ssim_scores),
            'mean_mse': np.mean(mse_scores),
            'std_mse': np.std(mse_scores),
            'min_ssim': np.min(ssim_scores),
            'max_ssim': np.max(ssim_scores)
        }
        
        return metrics

class VisualizationHelper:
    """Utility class for creating visualizations and plots"""
    
    @staticmethod
    def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # Loss plot
        if 'loss' in history:
            axes[0, 0].plot(history['loss'])
            axes[0, 0].set_title('Training Loss')
            axes[0, 0].set_xlabel('Step')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
        
        # Learning rate plot
        if 'learning_rate' in history:
            axes[0, 1].plot(history['learning_rate'])
            axes[0, 1].set_title('Learning Rate')
            axes[0, 1].set_xlabel('Step')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
        
        # Validation metrics
        if 'eval_loss' in history:
            axes[1, 0].plot(history['eval_loss'])
            axes[1, 0].set_title('Validation Loss')
            axes[1, 0].set_xlabel('Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True)
        
        # Gradient norm
        if 'grad_norm' in history:
            axes[1, 1].plot(history['grad_norm'])
            axes[1, 1].set_title('Gradient Norm')
            axes[1, 1].set_xlabel('Step')
            axes[1, 1].set_ylabel('Norm')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_image_comparison(original: Image.Image, generated: Image.Image, 
                             save_path: Optional[str] = None):
        """Plot original vs generated image comparison"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle('Original vs Generated Image Comparison', fontsize=16)
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Generated image
        axes[1].imshow(generated)
        axes[1].set_title('Generated Image')
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Image comparison plot saved to: {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_style_transfer_visualization(style_images: List[Image.Image], 
                                          generated_images: List[Image.Image],
                                          save_path: Optional[str] = None):
        """Create a visualization showing style transfer results"""
        n_images = min(len(style_images), len(generated_images))
        
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        fig.suptitle('Van Gogh Style Transfer Results', fontsize=16)
        
        for i in range(n_images):
            # Style reference
            axes[0, i].imshow(style_images[i])
            axes[0, i].set_title(f'Style Reference {i+1}')
            axes[0, i].axis('off')
            
            # Generated result
            axes[1, i].imshow(generated_images[i])
            axes[1, i].set_title(f'Generated Result {i+1}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Style transfer visualization saved to: {save_path}")
        
        plt.show()

class FileManager:
    """Utility class for file and directory management"""
    
    @staticmethod
    def ensure_directory(path: Union[str, Path]):
        """Ensure directory exists, create if it doesn't"""
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict, filepath: Union[str, Path]):
        """Save data to JSON file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Data saved to: {filepath}")
    
    @staticmethod
    def load_json(filepath: Union[str, Path]) -> Dict:
        """Load data from JSON file"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return {}
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        logger.info(f"Data loaded from: {filepath}")
        return data
    
    @staticmethod
    def get_file_size_mb(filepath: Union[str, Path]) -> float:
        """Get file size in MB"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            return 0.0
        
        size_bytes = filepath.stat().st_size
        return size_bytes / (1024 * 1024)
    
    @staticmethod
    def cleanup_old_files(directory: Union[str, Path], 
                         pattern: str = "*", 
                         max_age_days: int = 7):
        """Clean up old files in directory"""
        directory = Path(directory)
        
        if not directory.exists():
            return
        
        import time
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        for filepath in directory.glob(pattern):
            if filepath.is_file():
                file_age = current_time - filepath.stat().st_mtime
                if file_age > max_age_seconds:
                    filepath.unlink()
                    logger.info(f"Cleaned up old file: {filepath}")

def main():
    """Test utility functions"""
    print("Testing utility functions...")
    
    # Test image processing
    print("Testing image processing...")
    
    # Test file management
    print("Testing file management...")
    FileManager.ensure_directory("test_output")
    
    # Test data saving/loading
    test_data = {"test": "data", "number": 42}
    FileManager.save_json(test_data, "test_output/test.json")
    loaded_data = FileManager.load_json("test_output/test.json")
    print(f"Data round-trip test: {test_data == loaded_data}")
    
    print("Utility functions test completed!")

if __name__ == "__main__":
    main()
