"""Setup script for Native Language Identification package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="native-language-identification",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Native Language Identification of Indian English Speakers Using HuBERT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/native-language-identification",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.30.0",
        "librosa>=0.10.0",
        "soundfile>=0.11.0",
        "scikit-learn>=1.0.0",
        "datasets>=2.12.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "tensorboard>=2.10.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=5.0.0",
            "ipython>=8.0.0",
            "jupyter>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nli-train=src.training.train:main",
            "nli-eval=src.evaluation.evaluate:main",
        ],
    },
)
