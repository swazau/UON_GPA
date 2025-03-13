# GPA and WAM Calculator

A Python tool for visualising academic performance and calculating GPAs and WAMs from CSV grade data or PDF transcripts, designed for the University of Newcastle grading system.

## Features

### GPA Calculator
- 📊 Calculate overall GPA using a 7-point scale
- 📈 Visualise GPA trends across semesters
- 🥧 Generate grade distribution charts
- 📉 Analyse mark distributions
- 📋 View detailed course performance breakdowns
- 📄 Process UON transcripts directly from PDF files

### WAM Calculator
- 🎓 Calculate Honours WAM by default (2000+ level courses)
- 📊 Additional WAM calculations for reference:
  - Cumulative WAM (all courses)
  - Level-specific WAM (e.g., 3000+ level courses)
- 📈 Track WAM trends across semesters
- 📉 View mark distribution with WAM thresholds
- 📋 See detailed course-by-course breakdown
- 📄 Process UON transcripts directly from PDF files

## What is WAM?

A Weighted Average Mark (WAM) is the average mark achieved across all completed units in a program, weighted according to unit value and academic level. Unlike the GPA, which uses grade points, WAM uses the actual percentage marks and applies weighting based on course levels.

The Honours WAM (used for determining honours classification) is calculated using only 2000+ level courses with the following weightings:
- 2000 level courses: Weight = 2
- 3000 level courses: Weight = 3
- 4000/5000/6000 level courses: Weight = 4


## Screenshots

### GPA Trend
![GPA Trend Visualisation](screenshots/gpa_trend.png)

### Grade Distribution
![Grade Distribution](screenshots/grade_distribution.png)

### Mark Distribution
![Mark Distribution](screenshots/mark_distribution.png)

### Course Performance
![Course Performance](screenshots/course_performance.png)

### WAM Comparison
![WAM Comparison](screenshots/wam_comparison.png)

### WAM Trend
![WAM Trend](screenshots/wam_trend.png)

### WAM Honours Threshold
![Honours Threshold](screenshots/honours_thresholds.png)

## Installation

### Prerequisites
- Python 3.8 or higher
- `pip` (Python package installer)

### Option 1: Quick Setup (Recommended)

#### On Windows:
```bash
git clone https://github.com/swazau/UON_GPA.git
cd UON_GPA
setup.bat
```

#### On macOS/Linux:
```bash
git clone https://github.com/swazau/UON_GPA.git
cd UON_GPA
chmod +x setup.sh && ./setup.sh
```

### Option 2: Manual Setup
```bash
git clone https://github.com/swazau/UON_GPA.git
cd UON_GPA
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
Create a CSV file with your grade data, including:
- `semester`: e.g., '2023-S1'
- `course_code`: e.g., 'COMP1010'
- `grade`: HD, D, C, P, F, UP
- `mark`: 0-100
- `units`: Credit units
- `class_title`: Course name

Use `samplegrades.csv` as a template. Alternatively, provide an unofficial transcript as a PDF (e.g., `transcript.pdf`), and the program will extract the data using the `--pdf` flag. For best results, ensure the PDF follows the University of Newcastle format.

*Tip*: If you would pefer to provide your own csv, use ChatGPT to reformat your transcript into CSV by providing it with `samplegrades.csv` and your transcript PDF.

### Run the Programs

#### With The Sample CSV:
```bash
python gpa.py
python wam.py
```

#### With Custom CSV:
```bash
python gpa.py your_grades_file.csv
python wam.py your_grades_file.csv
```

#### With PDF Transcript:
```bash
python gpa.py transcript.pdf --pdf
python wam.py transcript.pdf --pdf
```
The `--pdf` flag uses `transcript_processor` to extract data and generate visualisations.

### Output

#### GPA Program Output
The program generates:
1. Detailed GPA calculations in the console
2. Four visualisation files:
   - `grade_distribution.png`: Pie chart showing grade distribution by units
   - `mark_distribution.png`: Histogram of numerical marks
   - `course_performance.png`: Bar chart of performance by course
   - `gpa_trend.png`: Line chart showing semester and cumulative GPA trends

#### WAM Program Output
The program generates:
1. Honours WAM calculation (2000+ level courses) prominently displayed
2. Additional WAM calculations for reference in the console
3. Three visualisation files:
   - `wam_comparison.png`: Bar chart comparing different WAM calculations
   - `wam_mark_distribution.png`: Histogram with WAM thresholds
   - `wam_trend.png`: Line chart showing semester and cumulative WAM trends
   - `honours_threshold.png`: The Honours Thresholds graph is a custom horizontal bar chart that visually displays where your calculated WAM

## Grade Scales

### GPA Scale
| Grade | Points | Description        |
|-------|---------|--------------------|
| HD    | 7       | High Distinction   |
| D     | 6       | Distinction        |
| C     | 5       | Credit             |
| P     | 4       | Pass               |
| F     | 0       | Fail               |

### WAM Mark Values
| Grade | Mark Range | Value for WAM |
|-------|------------|---------------|
| HD, D, C, P | 50-100 | Actual percentage mark |
| F | 45-49 | Actual percentage mark |
| F | 0-44 | Fixed value of 44 |
| UP | N/A | Fixed value of 58 |

## Customisation

You can modify the grade points in the `GPAVisualiser` class initialisation if your institution uses a different scale.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- [University of Newcastle GPA Calculator](https://www.newcastle.edu.au/current-students/study-essentials/assessment-and-exams/results/gpa-calculator)
- [University of Newcastle WAM Calculation Guideline](https://policies.newcastle.edu.au/document/view-current.php?id=3)
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/) for visualisation