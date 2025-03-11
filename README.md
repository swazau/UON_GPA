# GPA Visualiser

A Python tool for visualising academic performance and calculating GPAs from CSV grade data, designed for the University of Newcastle GPA grading system: https://www.newcastle.edu.au/current-students/study-essentials/assessment-and-exams/results/gpa-calculator

## Features

- 📊 Calculate overall GPA using a 7-point scale
- 📈 Visualise GPA trends across semesters
- 🥧 Generate grade distribution charts
- 📉 Analyse mark distributions
- 📋 View detailed course performance breakdowns

## Screenshots

### GPA Trend
![GPA Trend Visualisation](screenshots/gpa_trend.png)

### Grade Distribution
![Grade Distribution](screenshots/grade_distribution.png)

### Mark Distribution
![Mark Distribution](screenshots/mark_distribution.png)

### Course Performance
![Course Performance](screenshots/course_performance.png)

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Option 1: Quick Setup (Recommended)

#### On Windows:

```bash
# Clone the repository
git clone https://github.com/swazau/UON_GPA.git
cd UON_GPA

# Run the setup script
setup.bat
```

#### On macOS/Linux:

```bash
# Clone the repository
git clone https://github.com/swazau/UON_GPA.git
cd UON_GPA

# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/swazau/UON_GPA.git
cd UON_GPA

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Prepare Your Data

Create a CSV file with your grade data. The file should include the following columns:

- `semester`: Academic term (e.g., '2023-S1')
- `course_code`: Course identifier (e.g., 'COMP1010')
- `grade`: Letter grade (HD, D, C, P, F)
- `mark`: Numerical score (0-100)
- `units`: Credit units for the course
- `class_title`: Full name of the course

You can use `samplegrades.csv` as a template. The easiest way to do this is just put your unofficial transcript and my template into ChatGPT, and it will reformat it correctly.

### Run the Program

To run the program with the default `samplegrades.csv` file:

```bash
python main.py
```

To specify a different CSV file:

```bash
python main.py your_grades_file.csv
```

### Output

The program generates:
1. Detailed GPA calculations in the console
2. Four visualisation files:
   - `grade_distribution.png`: Pie chart showing grade distribution by units
   - `mark_distribution.png`: Histogram of numerical marks
   - `course_performance.png`: Bar chart of performance by course
   - `gpa_trend.png`: Line chart showing semester and cumulative GPA trends

## Grade Scale

This visualiser uses the following 7-point grade scale:

| Grade | Points | Description        |
|-------|---------|--------------------|
| HD    | 7       | High Distinction   |
| D     | 6       | Distinction        |
| C     | 5       | Credit             |
| P     | 4       | Pass               |
| F     | 0       | Fail               |

## Customisation

You can modify the grade points in the `GPAVisualiser` class initialisation if your institution uses a different scale.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualisation