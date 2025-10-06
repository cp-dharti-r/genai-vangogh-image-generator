# Makefile for Van Gogh Image Generator
.PHONY: help install test clean data train demo web all

# Default target
help:
	@echo "🎨 Van Gogh Image Generator - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "📦 Installation & Setup:"
	@echo "  install          Install all dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-gpu      Install GPU-optimized PyTorch"
	@echo ""
	@echo "🧪 Testing & Validation:"
	@echo "  test             Run installation tests"
	@echo "  test-imports     Test package imports only"
	@echo ""
	@echo "📊 Data & Training:"
	@echo "  data             Prepare Van Gogh dataset"
	@echo "  train            Run complete training pipeline"
	@echo "  train-skip-data  Train with existing dataset"
	@echo ""
	@echo "🎨 Generation & Demo:"
	@echo "  demo             Run demo with example prompts"
	@echo "  web              Launch web interface"
	@echo ""
	@echo "🔄 Complete Pipeline:"
	@echo "  all              Run complete pipeline (data + train + demo)"
	@echo ""
	@echo "🧹 Maintenance:"
	@echo "  clean            Clean generated files and outputs"
	@echo "  clean-data       Clean dataset files only"
	@echo "  clean-outputs    Clean training outputs only"
	@echo ""
	@echo "📋 Information:"
	@echo "  status           Show project status and file sizes"
	@echo "  logs             Show recent log files"

# Installation targets
install:
	@echo "📦 Installing dependencies..."
	pip install -r requirements.txt

install-dev:
	@echo "🔧 Installing development dependencies..."
	pip install -e .[dev]

install-gpu:
	@echo "🚀 Installing GPU-optimized PyTorch..."
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Testing targets
test:
	@echo "🧪 Running installation tests..."
	python test_installation.py

test-imports:
	@echo "🔍 Testing package imports..."
	python -c "import torch, transformers, diffusers, peft, gradio; print('✅ All imports successful')"

# Data and training targets
data:
	@echo "📊 Preparing Van Gogh dataset..."
	python data_preparation.py

train:
	@echo "🤖 Starting training pipeline..."
	python train.py

train-skip-data:
	@echo "🤖 Training with existing dataset..."
	python train.py --skip-data-prep

# Generation and demo targets
demo:
	@echo "🎨 Running demo..."
	python demo.py

web:
	@echo "🌐 Launching web interface..."
	python web_app.py

# Complete pipeline
all: data train demo
	@echo "🎉 Complete pipeline finished!"

# Cleanup targets
clean:
	@echo "🧹 Cleaning all generated files..."
	rm -rf data/processed data/raw data/metadata.json
	rm -rf outputs/
	rm -rf generated_images/
	rm -rf test_installation_output/
	rm -f *.log
	rm -f demo_*.png
	@echo "✅ Cleanup completed"

clean-data:
	@echo "🧹 Cleaning dataset files..."
	rm -rf data/processed data/raw data/metadata.json

clean-outputs:
	@echo "🧹 Cleaning training outputs..."
	rm -rf outputs/
	rm -rf generated_images/

# Information targets
status:
	@echo "📋 Project Status"
	@echo "================="
	@echo ""
	@echo "📁 Directory Structure:"
	@ls -la
	@echo ""
	@echo "📊 Dataset Status:"
	@if [ -d "data" ]; then \
		echo "Data directory exists"; \
		if [ -f "data/metadata.json" ]; then \
			echo "Dataset metadata found"; \
			echo "Processed images: $$(ls data/processed/*.png 2>/dev/null | wc -l)"; \
		else \
			echo "Dataset not prepared"; \
		fi; \
	else \
		echo "Data directory not found"; \
	fi
	@echo ""
	@echo "🤖 Model Status:"
	@if [ -d "outputs" ]; then \
		echo "Outputs directory exists"; \
		echo "Trained models: $$(ls outputs/*/ 2>/dev/null | wc -l)"; \
	else \
		echo "No trained models found"; \
	fi
	@echo ""
	@echo "💾 Disk Usage:"
	@du -sh . 2>/dev/null | head -1

logs:
	@echo "📋 Recent Log Files"
	@echo "==================="
	@if [ -f "training.log" ]; then \
		echo "Training log (last 10 lines):"; \
		tail -10 training.log; \
	else \
		echo "No training log found"; \
	fi

# Development targets
format:
	@echo "🎨 Formatting code..."
	black *.py
	@echo "✅ Code formatting completed"

lint:
	@echo "🔍 Running linter..."
	flake8 *.py
	@echo "✅ Linting completed"

# Quick start for new users
quickstart: install test data demo
	@echo ""
	@echo "🚀 Quick start completed!"
	@echo "Next steps:"
	@echo "  1. Run 'make train' to start fine-tuning"
	@echo "  2. Run 'make web' to launch the web interface"
	@echo "  3. Check 'make help' for more commands"

# Default target
.DEFAULT_GOAL := help
