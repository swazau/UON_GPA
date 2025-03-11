#!/bin/bash

# GPA Visualizer Setup Script for Unix/macOS

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed or not in PATH. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "==================================="
echo "Setup complete!"
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To run the GPA Visualizer:"
echo "python main.py"
echo "or"
echo "python main.py your_grades_file.csv"
echo "==================================="