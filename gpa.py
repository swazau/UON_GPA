import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import os


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

    def plot_grade_distribution(self, df: pd.DataFrame, output_path: Optional[str] = 'grade_distribution.png') -> None:
        """Create a pie chart of grade distribution by units"""
        plt.figure(figsize=(10, 6))
        valid_grades = df[df['grade'].isin(self.grade_points.keys())]
        grade_units = valid_grades.groupby('grade')['units'].sum()

        colors = sns.color_palette("Set2", len(grade_units))
        plt.pie(grade_units.values, labels=grade_units.index, autopct='%1.1f%%',
                colors=colors)
        plt.title('Grade Distribution (by Units)')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_mark_distribution(self, df: pd.DataFrame, output_path: Optional[str] = 'mark_distribution.png') -> None:
        """Create a histogram of numerical marks"""
        plt.figure(figsize=(10, 6))
        valid_marks = df[df['mark'].notna()]['mark']

        sns.histplot(data=valid_marks, bins=10, kde=True, color=sns.color_palette("husl")[2])
        plt.axvline(valid_marks.mean(), color='red', linestyle='--',
                    label=f'Mean: {valid_marks.mean():.1f}')

        plt.title('Distribution of Numerical Marks')
        plt.xlabel('Mark')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_course_performance(self, df: pd.DataFrame, output_path: Optional[str] = 'course_performance.png') -> None:
        """Create a horizontal bar chart of performance by course"""
        plt.figure(figsize=(12, 8))
        valid_courses = df[df['mark'].notna()].sort_values('mark', ascending=True)

        colors = {
            'HD': sns.color_palette("Set2")[0],
            'D': sns.color_palette("Set2")[1],
            'C': sns.color_palette("Set2")[2],
            'P': sns.color_palette("Set2")[3],
            'F': sns.color_palette("Set2")[4]
        }

        bars = plt.barh(valid_courses['course_code'], valid_courses['mark'])

        for bar, grade in zip(bars, valid_courses['grade']):
            if grade in colors:
                bar.set_color(colors[grade])

        plt.axvline(valid_courses['mark'].mean(), color='red', linestyle='--',
                    label=f'Mean: {valid_courses["mark"].mean():.1f}')

        plt.title('Performance by Course')
        plt.xlabel('Mark')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

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

    def plot_gpa_trend(self, df: pd.DataFrame, output_path: Optional[str] = 'gpa_trend.png') -> None:
        """
        Create a line plot showing GPA trend across semesters

        Parameters:
        df (pd.DataFrame): DataFrame containing grade data
        output_path (str, optional): Path to save the plot. Defaults to 'gpa_trend.png'.
        """
        # Calculate GPA for each semester
        semester_gpa = self.calculate_semester_gpa(df)

        # Create the figure
        plt.figure(figsize=(12, 6))

        # Plot the GPA trend line
        sns.lineplot(
            x='semester',
            y='gpa',
            data=semester_gpa,
            marker='o',
            markersize=10,
            linewidth=2,
            color=sns.color_palette("husl")[0]
        )

        # Calculate and plot cumulative GPA
        cumulative_points = 0
        cumulative_units = 0
        cumulative_gpa = []

        for _, row in semester_gpa.iterrows():
            cumulative_units += row['units']
            cumulative_points += row['gpa'] * row['units']
            if cumulative_units > 0:
                cumulative_gpa.append(round(cumulative_points / cumulative_units, 2))
            else:
                cumulative_gpa.append(0)

        plt.plot(
            range(len(semester_gpa)),
            cumulative_gpa,
            linestyle='--',
            marker='s',
            markersize=8,
            color=sns.color_palette("husl")[1],
            label='Cumulative GPA'
        )

        # Set plot properties
        plt.title('GPA Trend by Semester', fontsize=16)
        plt.xlabel('Semester', fontsize=12)
        plt.ylabel('GPA', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 7.5)  # Setting y-axis range based on the 7-point GPA scale
        plt.tight_layout()
        plt.legend(['Semester GPA', 'Cumulative GPA'])

        # Save the figure
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory


def main():
    visualiser = GPAVisualiser()

    # Get the CSV file path from command line arguments or use the default
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'samplegrades.csv'

    # Check if the CSV file exists and use it
    try:
        print(f"Attempting to load data from {csv_path}...")
        df = visualiser.load_from_csv(csv_path)
        print(f"Successfully loaded data from {csv_path}.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading CSV: {e}")
        return

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