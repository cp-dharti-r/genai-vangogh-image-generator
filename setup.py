#!/usr/bin/env python3
"""
Setup script for Van Gogh Image Generator
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open("requirements.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name="vangogh-image-generator",
    version="1.0.0",
    author="Van Gogh AI Team",
    author_email="contact@vangogh-ai.com",
    description="AI-powered image generation system fine-tuned for Van Gogh style using LoRA/QLoRA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/genai-vangogh-image-generator",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/genai-vangogh-image-generator/issues",
        "Source": "https://github.com/yourusername/genai-vangogh-image-generator",
        "Documentation": "https://github.com/yourusername/genai-vangogh-image-generator/wiki",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.15",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=0.15",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "vangogh-train=train:main",
            "vangogh-generate=image_generator:main",
            "vangogh-web=web_app:main",
            "vangogh-demo=demo:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.json", "*.yaml", "*.yml"],
    },
    keywords=[
        "ai", "art", "image-generation", "stable-diffusion", 
        "lora", "qlora", "fine-tuning", "vangogh", "style-transfer"
    ],
    platforms=["any"],
    license="MIT",
    zip_safe=False,
)
