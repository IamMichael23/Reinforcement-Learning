#!/bin/bash

# Setup script for Reinforcement Learning project with Python 3.10

echo "🚀 Setting up Python 3.10 virtual environment..."

# Create venv with Python 3.10
/Library/Frameworks/Python.framework/Versions/3.10/bin/python3.10 -m venv venv

# Activate venv
source venv/bin/activate

echo "📦 Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

echo "🛠 Configuring macOS SDK for native builds..."
export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
export CFLAGS="-isysroot $SDKROOT"
export CXXFLAGS="-isysroot $SDKROOT -I$SDKROOT/usr/include/c++/v1"
export ARCHFLAGS="-arch arm64"

echo "📥 Installing requirements..."
pip install \
  gym==0.26.2 \
  gym-super-mario-bros==7.4.0 \
  lz4>=4.3,<5.0 \
  nes-py==8.2.1 \
  numpy==1.26.4 \
  opencv-python==4.9.0.80 \
  torch>=2.3,<3.0 \
  tensordict==0.11.0 \
  torchrl==0.11.1

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run your code:"
echo "  python src/main.py"
