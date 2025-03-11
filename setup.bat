@echo off
REM GPA Visualizer Setup Script for Windows

REM Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    exit /b 1
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo ===================================
echo Setup complete!
echo To activate the environment in the future, run:
echo venv\Scripts\activate.bat
echo.
echo To run the GPA Visualizer:
echo python main.py
echo or
echo python main.py your_grades_file.csv
echo ===================================