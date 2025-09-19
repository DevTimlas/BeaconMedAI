import sys
import os
from setuptools import setup, find_packages

# Ensure src directory is in sys.path for relative imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

setup(
    name="BeaconMedAI",
    version="0.1.0",
    description="AMA-compliant impairment report and rebuttal generator",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "gradio>=3.50.0",
        "langchain>=0.0.300",
        "langchain-openai>=0.0.2",
        "langchain-community>=0.0.20",
        "openai>=1.0.0",
        "groq>=0.9.0",
        "huggingface_hub>=0.19.0",
        "PyMuPDF>=1.23.0",
        "chromadb>=0.4.0",
        "loguru>=0.7.0",
        "cryptography>=41.0.0",
        "tiktoken>=0.5.0",
        "scikit-learn>=1.3.0",
        "python-Levenshtein>=0.20.0",
        "numpy>=1.26.0",
        "python-dotenv>=1.0.0",
        "moviepy>=1.0.3",
        "markdown-pdf>=1.1.2",
        "Pillow>=10.0.0",
        "speechrecognition>=3.10.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)