#!/usr/bin/env python3
"""
Test script to verify Van Gogh Image Generator installation
Checks all dependencies and basic functionality
"""
import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "transformers",
        "diffusers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "datasets",
        "PIL",
        "cv2",
        "numpy",
        "matplotlib",
        "seaborn",
        "tqdm",
        "gradio"
    ]
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                importlib.import_module("PIL.Image")
            elif package == "cv2":
                importlib.import_module("cv2")
            else:
                importlib.import_module(package)
            print(f"  ✅ {package}")
        except ImportError as e:
            print(f"  ❌ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        return False
    else:
        print("✅ All packages imported successfully!")
        return True

def test_project_modules():
    """Test if project modules can be imported"""
    print("\n🔍 Testing project modules...")
    
    project_modules = [
        "config",
        "data_preparation", 
        "fine_tuning",
        "image_generator",
        "utils"
    ]
    
    failed_modules = []
    
    for module in project_modules:
        try:
            importlib.import_module(module)
            print(f"  ✅ {module}")
        except ImportError as e:
            print(f"  ❌ {module}: {e}")
            failed_modules.append(module)
    
    if failed_modules:
        print(f"\n❌ Failed to import modules: {', '.join(failed_modules)}")
        return False
    else:
        print("✅ All project modules imported successfully!")
        return True

def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import training_config, generation_config, web_config
        
        print(f"  ✅ Training config: {training_config.base_model}")
        print(f"  ✅ Generation config: {generation_config.num_inference_steps} steps")
        print(f"  ✅ Web config: {web_config.host}:{web_config.port}")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False

def test_utility_functions():
    """Test utility functions"""
    print("\n🔍 Testing utility functions...")
    
    try:
        from utils import FileManager, ImageProcessor
        
        # Test directory creation
        test_dir = "test_installation_output"
        FileManager.ensure_directory(test_dir)
        
        # Test JSON operations
        test_data = {"test": "data", "number": 42}
        FileManager.save_json(test_data, f"{test_dir}/test.json")
        loaded_data = FileManager.load_json(f"{test_dir}/test.json")
        
        if test_data == loaded_data:
            print("  ✅ FileManager functions working")
        else:
            print("  ❌ FileManager data mismatch")
            return False
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
    except Exception as e:
        print(f"  ❌ Utility functions error: {e}")
        return False

def test_hardware():
    """Test hardware availability"""
    print("\n🔍 Testing hardware...")
    
    try:
        import torch
        
        # Check CUDA availability
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA available: {device_count} device(s)")
            print(f"  ✅ Primary device: {device_name}")
        else:
            print("  ℹ️ CUDA not available, will use CPU")
        
        # Check PyTorch version
        print(f"  ✅ PyTorch version: {torch.__version__}")
        
        return True
    except Exception as e:
        print(f"  ❌ Hardware test error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎨 Van Gogh Image Generator - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Package Imports", test_imports),
        ("Project Modules", test_project_modules),
        ("Configuration", test_configuration),
        ("Utility Functions", test_utility_functions),
        ("Hardware", test_hardware)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Installation is successful.")
        print("\n🚀 You can now:")
        print("  1. Run 'python data_preparation.py' to prepare the dataset")
        print("  2. Run 'python train.py' to start fine-tuning")
        print("  3. Run 'python web_app.py' to launch the web interface")
        print("  4. Run 'python demo.py' to see example generations")
        return True
    else:
        print(f"❌ {total - passed} test(s) failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
