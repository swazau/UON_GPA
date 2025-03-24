import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import os
import argparse
import plotly.express as px

class GPAVisualiser:
    def __init__(self):
        self.grade_points = {
            'HD': 7,
            'D': 6,
            'C': 5,
            'P': 4,
            'F': 0
        }
        plt.style.use('seaborn-v0_8')
        sns.set_theme()

    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load grade data from a CSV file

        Parameters:
        csv_path (str): Path to the CSV file

        Returns:
        pd.DataFrame: DataFrame containing the grade data
        """
        # Check if file exists
        file_path = Path(csv_path)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Read CSV file
        df = pd.read_csv(csv_path)

        # Validate required columns - removed class_title from required columns
        required_columns = ['semester', 'course_code', 'grade', 'mark', 'units']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {', '.join(missing_columns)}")

        # Convert 'mark' column to float, handling potential missing values
        df['mark'] = pd.to_numeric(df['mark'], errors='coerce')

        # Convert 'units' column to int, handling potential missing values
        df['units'] = pd.to_numeric(df['units'], errors='coerce').fillna(0).astype(int)

        return df

    def calculate_university_gpa(self, df: pd.DataFrame) -> Dict:
        """Calculate GPA using the university's method"""
        # Filter out NA grades
        valid_grades = df[df['grade'].isin(self.grade_points.keys())]

        # Calculate total units for each grade
        grade_units = valid_grades.groupby('grade')['units'].sum()

        # Calculate total points for each grade category
        grade_points = {grade: units * self.grade_points[grade]
                        for grade, units in grade_units.items()}

        total_points = sum(grade_points.values())
        total_units = sum(grade_units)
        gpa = round(total_points / total_units, 2)

        return {
            'grade_units': dict(grade_units),
            'grade_points': grade_points,
            'total_points': total_points,
            'total_units': total_units,
            'gpa': gpa
        }

    def plot_grade_distribution(self, df: pd.DataFrame) -> 'plotly.graph_objs.Figure':
        valid_grades = df[df['grade'].isin(self.grade_points.keys())]
        grade_units = valid_grades.groupby('grade')['units'].sum().reset_index()
        fig = px.pie(grade_units, values='units', names='grade', title='Grade Distribution (by Units)',
                    color_discrete_sequence=px.colors.qualitative.Set2)
        return fig

    def plot_mark_distribution(self, df: pd.DataFrame) -> 'plotly.graph_objs.Figure':
        valid_marks = df[df['mark'].notna()]['mark']
        fig = px.histogram(valid_marks, nbins=10, title='Distribution of Numerical Marks')
        fig.update_traces(marker_color=px.colors.qualitative.Set2[2])
        mean_mark = valid_marks.mean()
        fig.add_vline(x=mean_mark, line_dash="dash", line_color="red",
                    annotation_text=f"Mean: {mean_mark:.1f}", annotation_position="top right")
        fig.update_layout(xaxis_title='Mark', yaxis_title='Count')
        return fig

    def plot_course_performance(self, df: pd.DataFrame) -> 'plotly.graph_objs.Figure':
        valid_courses = df[df['mark'].notna()].sort_values('mark', ascending=True)
        color_map = {
            'HD': px.colors.qualitative.Set2[0],
            'D': px.colors.qualitative.Set2[1],
            'C': px.colors.qualitative.Set2[2],
            'P': px.colors.qualitative.Set2[3],
            'F': px.colors.qualitative.Set2[4]
        }
        fig = px.bar(valid_courses, y='course_code', x='mark', orientation='h',
                    color='grade', color_discrete_map=color_map,
                    title='Performance by Course')
        mean_mark = valid_courses['mark'].mean()
        fig.add_vline(x=mean_mark, line_dash="dash", line_color="red",
                    annotation_text=f"Mean: {mean_mark:.1f}", annotation_position="top left")
        fig.update_layout(xaxis_title='Mark', yaxis_title='Course Code')
        return fig

    def calculate_semester_gpa(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate GPA for each semester

        Parameters:
        df (pd.DataFrame): DataFrame containing grade data

        Returns:
        pd.DataFrame: DataFrame containing semester, GPA, and units
        """
        # Filter out NA grades and courses with no units
        valid_grades = df[(df['grade'].isin(self.grade_points.keys())) & (df['units'] > 0)]

        # Create a DataFrame to hold semester GPAs
        semester_gpa = []

        # Get unique semesters in chronological order
        semesters = sorted(valid_grades['semester'].unique())

        # Calculate GPA for each semester
        for semester in semesters:
            semester_df = valid_grades[valid_grades['semester'] == semester]

            # Calculate total points and units for the semester
            semester_points = 0
            semester_units = 0

            for _, row in semester_df.iterrows():
                grade_points = self.grade_points[row['grade']]
                units = row['units']
                semester_points += grade_points * units
                semester_units += units

            # Calculate semester GPA
            if semester_units > 0:
                gpa = round(semester_points / semester_units, 2)
            else:
                gpa = 0

            semester_gpa.append({
                'semester': semester,
                'gpa': gpa,
                'units': semester_units
            })

        return pd.DataFrame(semester_gpa)

    def plot_gpa_trend(self, df: pd.DataFrame) -> 'plotly.graph_objs.Figure':
        semester_gpa = self.calculate_semester_gpa(df)
        fig = px.line(semester_gpa, x='semester', y='gpa', markers=True, title='GPA Trend by Semester')
        fig.update_traces(line_color=px.colors.qualitative.Set2[0], marker_size=10)
        # Assume cumulative_gpa is calculated as a list or Series
        cumulative_gpa = semester_gpa['gpa'].cumsum() / (semester_gpa.index + 1)  # Example calculation
        fig.add_scatter(x=semester_gpa['semester'], y=cumulative_gpa, mode='lines+markers',
                        line_dash='dash', marker_symbol='square', marker_size=8,
                        line_color=px.colors.qualitative.Set2[1], name='Cumulative GPA')
        fig.update_layout(yaxis_range=[0, 7.5], xaxis_title='Semester', yaxis_title='GPA')
        return fig


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory


def main():
    #visualiser = GPAVisualiser()

    try:
        import transcript_processor
    except ImportError:
        print("Warning: transcript_processor module not found. PDF import functionality will be disabled.")
        transcript_processor = None

    parser = argparse.ArgumentParser(description='Calculate GPA from grade data')
    parser.add_argument('file', nargs='?', help='CSV or PDF file containing grade data')
    parser.add_argument('--pdf', action='store_true', help='Process input as PDF transcript')
    args = parser.parse_args()

    visualiser = GPAVisualiser()
    csv_path = 'samplegrades.csv'  # Default

    # Handle user-specified file
    if args.file:
        if args.pdf:
            # Process PDF transcript if --pdf flag is set
            print("pdf")
            if transcript_processor:
                print(f"Processing PDF transcript: {args.file}")
                csv_path = transcript_processor.process_transcript(args.file)
                if not csv_path:
                    print("Error processing transcript. Using default sample data.")
                    csv_path = 'samplegrades.csv'
            else:
                print("Transcript processor not available. Please install pdfplumber package.")
                return
        else:
            # Assume it's a CSV file
            csv_path = args.file

    # Load data and continue with existing GPA calculation
    try:
        df = visualiser.load_from_csv(csv_path)
        print("Successfully loaded grade data.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading CSV: {e}")
        print("Falling back to built-in data.")
        df = visualiser.create_grade_data()


    # Get the CSV file path from command line arguments or use the default
    # import sys
    # csv_path = sys.argv[1] if len(sys.argv) > 1 else 'samplegrades.csv'
    #
    # # Check if the CSV file exists and use it
    # try:
    #     print(f"Attempting to load data from {csv_path}...")
    #     df = visualiser.load_from_csv(csv_path)
    #     print(f"Successfully loaded data from {csv_path}.")
    # except (FileNotFoundError, ValueError) as e:
    #     print(f"Error loading CSV: {e}")
    #     return

    # Calculate GPA using university method
    results = visualiser.calculate_university_gpa(df)

    # Print detailed analysis
    print("\nGPA Calculation Details")
    print("=" * 50)
    print(f"\nTotal units completed: {results['total_units']}")
    print("\nBreakdown by grade:")
    print("-" * 30)
    print(f"{'Grade':<6} {'Units':>8} {'Points':>8}")
    print("-" * 30)

    for grade in ['HD', 'D', 'C', 'P', 'F']:
        if grade in results['grade_units']:
            units = results['grade_units'][grade]
            points = results['grade_points'][grade]
            print(f"{grade:<6} {units:>8} {points:>8}")

    print("-" * 30)
    print(f"{'Total':<6} {results['total_units']:>8} {results['total_points']:>8}")
    print(f"\nGPA: {results['gpa']}")

    gpa_dir = ensure_dir("GPA")

    # Generate visualisations
    visualiser.plot_grade_distribution(df, os.path.join(gpa_dir, 'grade_distribution.png'))
    visualiser.plot_mark_distribution(df, os.path.join(gpa_dir, 'mark_distribution.png'))
    visualiser.plot_course_performance(df, os.path.join(gpa_dir, 'course_performance.png'))
    visualiser.plot_gpa_trend(df, os.path.join(gpa_dir, 'gpa_trend.png'))

    # Print semester GPA information
    semester_gpa_df = visualiser.calculate_semester_gpa(df)
    print("\nSemester GPA Breakdown:")
    print("-" * 40)
    print(f"{'Semester':<10} {'Units':>8} {'GPA':>8}")
    print("-" * 40)
    for _, row in semester_gpa_df.iterrows():
        print(f"{row['semester']:<10} {row['units']:>8} {row['gpa']:>8.2f}")

    # Calculate mark statistics
    valid_marks = df[df['mark'].notna()]
    print(f"\nMark Statistics:")
    print(f"Average Mark: {valid_marks['mark'].mean():.2f}")
    print(f"Highest Mark: {valid_marks['mark'].max():.2f}")
    print(f"Lowest Mark: {valid_marks['mark'].min():.2f}")


if __name__ == "__main__":
    main()