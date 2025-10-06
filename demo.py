#!/usr/bin/env python3
"""
Demo script for Van Gogh Image Generator
Showcases the system's capabilities with example prompts
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from image_generator import VanGoghImageGenerator
from utils import ImageProcessor, VisualizationHelper
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_demo():
    """Run the Van Gogh Image Generator demo"""
    print("🎨 Van Gogh Image Generator - Demo")
    print("=" * 50)
    
    # Initialize generator
    print("Initializing image generator...")
    generator = VanGoghImageGenerator()
    
    # Example prompts for Van Gogh style
    demo_prompts = [
        "a sunflower field at golden hour",
        "a starry night over a small village",
        "a wheat field with cypress trees",
        "a self-portrait with swirling background",
        "a café terrace at night with warm lights",
        "an almond blossom tree in spring",
        "a bedroom with simple furniture",
        "a red vineyard in autumn"
    ]
    
    print(f"Running demo with {len(demo_prompts)} example prompts...")
    print("This may take several minutes depending on your hardware...")
    
    generated_images = []
    
    for i, prompt in enumerate(demo_prompts, 1):
        print(f"\n🎨 Generating image {i}/{len(demo_prompts)}: '{prompt}'")
        
        try:
            # Generate image
            image = generator.generate_image(
                prompt=prompt,
                num_steps=30,  # Reduced for demo speed
                guidance_scale=7.5
            )
            
            if image:
                # Apply Van Gogh style enhancement
                enhanced_image = ImageProcessor.apply_van_gogh_filter(image)
                
                # Save both original and enhanced
                original_path = generator.save_image(
                    image, 
                    f"demo_{i:02d}_original_{prompt.replace(' ', '_')[:20]}.png"
                )
                enhanced_path = generator.save_image(
                    enhanced_image, 
                    f"demo_{i:02d}_enhanced_{prompt.replace(' ', '_')[:20]}.png"
                )
                
                generated_images.append({
                    'prompt': prompt,
                    'original': image,
                    'enhanced': enhanced_image,
                    'original_path': original_path,
                    'enhanced_path': enhanced_path
                })
                
                print(f"✅ Generated successfully!")
                print(f"   Original: {original_path}")
                print(f"   Enhanced: {enhanced_path}")
                
            else:
                print(f"❌ Failed to generate image for: {prompt}")
                
        except Exception as e:
            print(f"❌ Error generating image: {e}")
            continue
    
    # Create visualizations
    if generated_images:
        print(f"\n📊 Creating visualizations...")
        
        # Create style transfer visualization
        original_images = [item['original'] for item in generated_images[:4]]  # First 4
        enhanced_images = [item['enhanced'] for item in generated_images[:4]]
        
        viz_path = "demo_style_transfer_results.png"
        VisualizationHelper.create_style_transfer_visualization(
            original_images, 
            enhanced_images,
            save_path=viz_path
        )
        print(f"   Style transfer visualization: {viz_path}")
        
        # Create image grid
        grid_image = ImageProcessor.create_image_grid(
            [item['enhanced'] for item in generated_images], 
            cols=2
        )
        if grid_image:
            grid_path = generator.save_image(grid_image, "demo_all_images_grid.png")
            print(f"   Image grid: {grid_path}")
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"Generated {len(generated_images)} images in Van Gogh style")
        print(f"Check the 'generated_images' directory for results")
        
        # Show summary
        print(f"\n📋 Generated Images Summary:")
        for i, item in enumerate(generated_images, 1):
            print(f"   {i:2d}. {item['prompt']}")
            print(f"       Original: {Path(item['original_path']).name}")
            print(f"       Enhanced: {Path(item['enhanced_path']).name}")
    
    else:
        print("❌ No images were generated successfully")
        return False
    
    return True

def main():
    """Main demo function"""
    try:
        success = run_demo()
        if success:
            print("\n🚀 Demo completed! Try the web interface with:")
            print("   python web_app.py")
        else:
            print("\n❌ Demo failed. Check the logs for details.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
