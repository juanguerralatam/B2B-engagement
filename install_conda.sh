#!/bin/bash
# Installation script for B2B Engagement Analysis dependencies
# For conda base environment

echo "🚀 Installing B2B Engagement Analysis dependencies..."
echo "📍 Using conda base environment"

# Update conda first
echo "📦 Updating conda..."
conda update -n base -c defaults conda -y

# Install major packages via conda for better dependency resolution
echo "🔧 Installing core packages via conda..."

# PyTorch ecosystem (latest versions)
echo "⚡ Installing PyTorch 2.4+..."
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
# Remove 'cpuonly' and add 'pytorch-cuda=11.8' or 'pytorch-cuda=12.1' if you have CUDA

# Scientific computing stack
echo "📊 Installing scientific computing packages..."
conda install numpy=2.0 pandas=2.2 matplotlib=3.9 seaborn=0.13 scipy -c conda-forge -y

# Computer vision
echo "👁️ Installing computer vision packages..."
conda install opencv=4.10 pillow=10.4 -c conda-forge -y

# Jupyter
echo "📓 Installing Jupyter..."
conda install jupyter ipykernel -c conda-forge -y

# Audio processing
echo "🎵 Installing audio processing..."
conda install librosa=0.10 -c conda-forge -y

# Basic utilities
echo "🛠️ Installing basic utilities..."
conda install requests psutil -c conda-forge -y

# Install remaining packages via pip
echo "📦 Installing remaining packages via pip..."
pip install --upgrade selenium>=4.23.0
pip install --upgrade webdriver-manager>=4.0.2
pip install --upgrade yt-dlp>=2024.8.6
pip install --upgrade moviepy>=1.0.3
pip install --upgrade openai-whisper>=20240930
pip install --upgrade "scenedetect[opencv]>=0.6.4"
pip install --upgrade google-api-python-client>=2.140.0
pip install --upgrade isodate>=0.6.1
pip install --upgrade deepface>=0.0.93
pip install --upgrade tensorflow>=2.17.0
pip install --upgrade facenet-pytorch>=2.6.0
pip install --upgrade timm>=1.0.8
pip install --upgrade opensmile>=2.6.0
pip install --upgrade nltk>=3.9.1
pip install --upgrade textblob>=0.18.0
pip install --upgrade vaderSentiment>=3.3.2
pip install --upgrade transformers>=4.44.0
pip install --upgrade tokenizers>=0.19.1
pip install --upgrade plotly>=5.23.0
pip install --upgrade python-dotenv>=1.0.1
pip install --upgrade browser-cookie3>=0.19.1
pip install --upgrade chardet>=5.0.0

echo "✅ Installation complete!"
echo "🎯 Your B2B Engagement Analysis environment is ready!"
echo ""
echo "📝 To verify installation, run:"
echo "   python -c 'import torch; print(f\"PyTorch version: {torch.__version__}\")'"
echo "   python -c 'import numpy; print(f\"NumPy version: {numpy.__version__}\")'"
