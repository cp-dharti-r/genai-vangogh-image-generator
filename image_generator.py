"""
Image generation module for Van Gogh Image Generator
Uses fine-tuned models to generate Van Gogh style images
"""
import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch
from PIL import Image
import numpy as np
from diffusers import (
    StableDiffusionPipeline,
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler
)
from peft import PeftModel
import gradio as gr

from config import generation_config, training_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VanGoghImageGenerator:
    """Generates Van Gogh style images using fine-tuned models"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or f"outputs/{training_config.model_name}"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline = None
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
    
    def load_model(self) -> bool:
        """Load the fine-tuned model"""
        try:
            logger.info("Loading fine-tuned model...")
            
            # Check if fine-tuned model exists
            if not Path(self.model_path).exists():
                logger.warning(f"Fine-tuned model not found at {self.model_path}")
                logger.info("Using base model instead")
                return self.load_base_model()
            
            # Load fine-tuned model
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            # Enable attention slicing for memory efficiency
            self.pipeline.enable_attention_slicing()
            
            logger.info("Fine-tuned model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            logger.info("Falling back to base model")
            return self.load_base_model()
    
    def load_base_model(self) -> bool:
        """Load the base Stable Diffusion model"""
        try:
            logger.info("Loading base Stable Diffusion model...")
            
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                training_config.base_model,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Move to device
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable memory efficient attention if available
            if hasattr(self.pipeline, "enable_xformers_memory_efficient_attention"):
                self.pipeline.enable_xformers_memory_efficient_attention()
            
            # Enable attention slicing for memory efficiency
            self.pipeline.enable_attention_slicing()
            
            logger.info("Base model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            return False
    
    def generate_image(
        self, 
        prompt: str, 
        negative_prompt: str = "",
        num_steps: int = None,
        guidance_scale: float = None,
        seed: Optional[int] = None
    ) -> Optional[Image.Image]:
        """Generate a single image"""
        if self.pipeline is None:
            if not self.load_model():
                return None
        
        # Use default values if not specified
        num_steps = num_steps or generation_config.num_inference_steps
        guidance_scale = guidance_scale or generation_config.guidance_scale
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        try:
            # Add Van Gogh style to prompt
            enhanced_prompt = f"{prompt}, {generation_config.style_prompt}"
            
            # Set default negative prompt if not provided
            if not negative_prompt:
                negative_prompt = "low quality, blurry, distorted, ugly, bad anatomy, watermark, signature"
            
            logger.info(f"Generating image with prompt: {enhanced_prompt}")
            
            # Generate image
            result = self.pipeline(
                prompt=enhanced_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                num_images_per_prompt=1
            )
            
            # Get the generated image
            image = result.images[0]
            
            logger.info("Image generated successfully")
            return image
            
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None
    
    def generate_multiple_images(
        self, 
        prompt: str, 
        num_images: int = 4,
        **kwargs
    ) -> List[Image.Image]:
        """Generate multiple images with the same prompt"""
        images = []
        
        for i in range(num_images):
            # Use different seed for each image
            seed = kwargs.get('seed', None)
            if seed is not None:
                seed += i
            
            image = self.generate_image(prompt, seed=seed, **kwargs)
            if image:
                images.append(image)
        
        return images
    
    def save_image(self, image: Image.Image, filename: str, output_dir: str = "generated_images"):
        """Save generated image to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_path = output_path / filename
        image.save(file_path, format=generation_config.output_format, quality=generation_config.output_quality)
        
        logger.info(f"Image saved to: {file_path}")
        return str(file_path)

class VanGoghWebInterface:
    """Web interface for Van Gogh Image Generator"""
    
    def __init__(self):
        self.generator = VanGoghImageGenerator()
    
    def create_interface(self):
        """Create the Gradio web interface"""
        
        def generate_single_image(prompt, negative_prompt, num_steps, guidance_scale, seed):
            """Generate a single image"""
            if not prompt.strip():
                return None, "Please enter a prompt"
            
            image = self.generator.generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            if image:
                return image, "Image generated successfully!"
            else:
                return None, "Failed to generate image"
        
        def generate_grid(prompt, negative_prompt, num_steps, guidance_scale, seed):
            """Generate a 2x2 grid of images"""
            if not prompt.strip():
                return None, "Please enter a prompt"
            
            images = self.generator.generate_multiple_images(
                prompt=prompt,
                num_images=4,
                negative_prompt=negative_prompt,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            if len(images) == 4:
                # Create 2x2 grid
                grid_width = images[0].width
                grid_height = images[0].height
                
                grid_image = Image.new('RGB', (grid_width * 2, grid_height * 2))
                
                for i, img in enumerate(images):
                    x = (i % 2) * grid_width
                    y = (i // 2) * grid_height
                    grid_image.paste(img, (x, y))
                
                return grid_image, f"Generated {len(images)} images successfully!"
            else:
                return None, f"Failed to generate images. Only {len(images)} generated."
        
        # Create the interface
        with gr.Blocks(title="Van Gogh Image Generator", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎨 Van Gogh Image Generator")
            gr.Markdown("Generate beautiful images in the style of Vincent van Gogh using AI fine-tuning techniques (LoRA/QLoRA)")
            
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column():
                        prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe what you want to see...",
                            lines=3
                        )
                        negative_prompt_input = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What you don't want to see...",
                            value="low quality, blurry, distorted, ugly, bad anatomy",
                            lines=2
                        )
                        
                        with gr.Row():
                            num_steps_input = gr.Slider(
                                minimum=20, maximum=100, value=50, step=1,
                                label="Number of Steps"
                            )
                            guidance_scale_input = gr.Slider(
                                minimum=1.0, maximum=20.0, value=7.5, step=0.1,
                                label="Guidance Scale"
                            )
                        
                        seed_input = gr.Number(
                            label="Seed (optional, for reproducibility)",
                            value=None
                        )
                        
                        generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                    
                    with gr.Column():
                        output_image = gr.Image(label="Generated Image")
                        output_text = gr.Textbox(label="Status", interactive=False)
                
                generate_btn.click(
                    fn=generate_single_image,
                    inputs=[prompt_input, negative_prompt_input, num_steps_input, guidance_scale_input, seed_input],
                    outputs=[output_image, output_text]
                )
            
            with gr.Tab("Image Grid"):
                with gr.Row():
                    with gr.Column():
                        grid_prompt_input = gr.Textbox(
                            label="Prompt",
                            placeholder="Describe what you want to see...",
                            lines=3
                        )
                        grid_negative_prompt_input = gr.Textbox(
                            label="Negative Prompt",
                            placeholder="What you don't want to see...",
                            value="low quality, blurry, distorted, ugly, bad anatomy",
                            lines=2
                        )
                        
                        with gr.Row():
                            grid_num_steps_input = gr.Slider(
                                minimum=20, maximum=100, value=50, step=1,
                                label="Number of Steps"
                            )
                            grid_guidance_scale_input = gr.Slider(
                                minimum=1.0, maximum=20.0, value=7.5, step=0.1,
                                label="Guidance Scale"
                            )
                        
                        grid_seed_input = gr.Number(
                            label="Seed (optional, for reproducibility)",
                            value=None
                        )
                        
                        generate_grid_btn = gr.Button("🎨 Generate Image Grid", variant="primary")
                    
                    with gr.Column():
                        grid_output_image = gr.Image(label="Generated Image Grid")
                        grid_output_text = gr.Textbox(label="Status", interactive=False)
                
                generate_grid_btn.click(
                    fn=generate_grid,
                    inputs=[grid_prompt_input, grid_negative_prompt_input, grid_num_steps_input, grid_guidance_scale_input, grid_seed_input],
                    outputs=[grid_output_image, grid_output_text]
                )
            
            with gr.Tab("About"):
                gr.Markdown("""
                ## About Van Gogh Image Generator
                
                This application uses advanced AI fine-tuning techniques to generate images in the style of Vincent van Gogh:
                
                - **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning method that adds small trainable matrices to the base model
                - **QLoRA (Quantized LoRA)**: Memory-efficient version that quantizes the base model to 4-bit precision
                
                ### How it works:
                1. **Data Collection**: High-quality Van Gogh paintings are collected and preprocessed
                2. **Fine-tuning**: The base Stable Diffusion model is fine-tuned using LoRA/QLoRA techniques
                3. **Generation**: Users can generate new images by providing text prompts
                
                ### Tips for better results:
                - Be descriptive in your prompts
                - Use art-related terms like "oil painting", "impressionist", "artistic"
                - Experiment with different guidance scales and step counts
                - Try different seeds for variety
                
                ### Technical Details:
                - Base Model: Stable Diffusion v1.5
                - Fine-tuning: LoRA with rank 16, alpha 32
                - Image Resolution: 512x512 pixels
                - Training: 100 epochs with learning rate 1e-4
                """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the web interface"""
        interface = self.create_interface()
        return interface.launch(**kwargs)

def main():
    """Main function for image generation"""
    # Test image generation
    generator = VanGoghImageGenerator()
    
    # Test prompt
    test_prompt = "a beautiful sunflower field at sunset"
    
    print(f"Generating image with prompt: {test_prompt}")
    
    image = generator.generate_image(test_prompt)
    if image:
        # Save the image
        filename = f"test_generation_{test_prompt.replace(' ', '_')[:20]}.png"
        output_path = generator.save_image(image, filename)
        print(f"Image saved to: {output_path}")
        
        # Show image info
        print(f"Image size: {image.size}")
        print(f"Image mode: {image.mode}")
    else:
        print("Failed to generate image")

if __name__ == "__main__":
    main()
