import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional


class WAMCalculator:
    def __init__(self):
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

        # Validate required columns
        required_columns = ['semester', 'course_code', 'grade', 'mark', 'units', 'class_title']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"CSV file missing required columns: {', '.join(missing_columns)}")

        # Convert 'mark' column to float, handling potential missing values
        df['mark'] = pd.to_numeric(df['mark'], errors='coerce')

        # Convert 'units' column to int, handling potential missing values
        df['units'] = pd.to_numeric(df['units'], errors='coerce').fillna(0).astype(int)

        return df

    def calculate_wam(self, df: pd.DataFrame, calculation_type='cumulative', specific_level=None,
                      specific_major=None) -> Dict:
        """
        Calculate Weighted Average Mark (WAM) based on the provided guidelines.

        Parameters:
        df (pd.DataFrame): DataFrame containing grade data
        calculation_type (str): Type of WAM calculation ('cumulative', 'level', 'major', 'annual')
        specific_level (int, optional): Specific level to include (e.g., 2000 for 2000+ level courses)
        specific_major (str, optional): Specific major to calculate WAM for

        Returns:
        Dict: Dictionary containing WAM calculation details
        """
        # Make a copy of the dataframe to avoid modifying original
        df_wam = df.copy()

        # Ensure the course_code column can be used to extract level information
        if not all(df_wam['course_code'].str.contains(r'\d{4}', regex=True)):
            raise ValueError("Course codes must follow the pattern with 4 digits (e.g., INFO1010)")

        # Extract course levels from course codes (assuming first digit is the level)
        df_wam['course_level'] = df_wam['course_code'].str.extract(r'([0-9])')[0].astype(int) * 1000

        # Filter based on calculation type
        if calculation_type == 'level' and specific_level is not None:
            df_wam = df_wam[df_wam['course_level'] >= specific_level]
        elif calculation_type == 'major' and specific_major is not None:
            # This would require additional data about which courses belong to which major
            # For now, just filtering by specific_major in course_code as a placeholder
            df_wam = df_wam[df_wam['course_code'].str.contains(specific_major)]
        elif calculation_type == 'annual':
            # This would require a year field, which we don't have in the current data model
            # For demonstration, using semester as a proxy (would need adjustment)
            if 'year' in df_wam.columns:
                year = df_wam['year'].max()  # Using the most recent year as an example
                df_wam = df_wam[df_wam['year'] == year]
            else:
                print("Warning: 'annual' calculation type needs a 'year' column. Using all data.")

        # Apply the WAM calculation rules according to the guidelines

        # Define the weighting based on course level
        level_weights = {
            1000: 1,
            2000: 2,
            3000: 3,
            4000: 4,
            5000: 4,
            6000: 4
        }

        # Apply the weights
        df_wam['weight'] = df_wam['course_level'].apply(lambda x: level_weights.get(x, 1))

        # Transform grades and marks according to WAM rules
        def map_mark_for_wam(row):
            # For UP (ungraded pass) use 58 as the mark value
            if row['grade'] == 'UP':
                return 58

            # For passing grades (P, C, D, HD), use the actual mark
            if row['grade'] in ['P', 'C', 'D', 'HD']:
                return row['mark']

            # For failing grades (45-49), use the actual mark
            if row['grade'] == 'F' and 45 <= row['mark'] <= 49:
                return row['mark']

            # For failing grades (0-44), use 44
            if row['grade'] == 'F' and row['mark'] < 45:
                return 44

            # Default case
            return row['mark']

        df_wam['wam_mark'] = df_wam.apply(map_mark_for_wam, axis=1)

        # Check for UP grades and apply special handling
        up_grades_count = df_wam[df_wam['grade'] == 'UP'].shape[0]
        total_grades_count = df_wam.shape[0]
        include_up_grades = False

        # Calculate preliminary WAM without UP grades
        df_without_up = df_wam[df_wam['grade'] != 'UP']

        # Calculation: WAM = Σ (M × V × W) / Σ (V × W)
        sum_weighted_marks = (df_without_up['wam_mark'] * df_without_up['units'] * df_without_up['weight']).sum()
        sum_weighted_units = (df_without_up['units'] * df_without_up['weight']).sum()

        if sum_weighted_units > 0:
            preliminary_wam = sum_weighted_marks / sum_weighted_units
        else:
            preliminary_wam = 0

        # Check if UP grades should be included
        if up_grades_count > 0.5 * total_grades_count:  # More than 50% are UP grades
            include_up_grades = True
        elif preliminary_wam < 58:  # WAM < 58, including UP grades will help
            include_up_grades = True

        # Final WAM calculation
        if include_up_grades:
            sum_weighted_marks = (df_wam['wam_mark'] * df_wam['units'] * df_wam['weight']).sum()
            sum_weighted_units = (df_wam['units'] * df_wam['weight']).sum()

        if sum_weighted_units > 0:
            final_wam = sum_weighted_marks / sum_weighted_units
        else:
            final_wam = 0

        # Round WAM according to rounding rules
        def round_wam(wam):
            # Round to 2 decimal places first
            wam_2dp = round(wam, 2)
            # Apply rounding rules
            decimal_part = wam_2dp - int(wam_2dp)
            if decimal_part < 0.5:
                return int(wam_2dp)
            else:
                return int(wam_2dp) + 1

        rounded_wam = round_wam(final_wam)

        # Prepare result dictionary
        result = {
            'calculation_type': calculation_type,
            'included_courses': len(df_wam),
            'total_units': df_wam['units'].sum(),
            'up_grades_included': include_up_grades,
            'raw_wam': final_wam,
            'rounded_wam': rounded_wam,
            'course_level_counts': df_wam.groupby('course_level')['units'].sum().to_dict()
        }

        return result

    def plot_wam_comparison(self, df: pd.DataFrame, output_path: str = 'wam_comparison.png') -> None:
        """
        Create a bar chart comparing WAM calculations with different criteria,
        highlighting the Honours WAM (2000+ level courses)

        Parameters:
        df (pd.DataFrame): DataFrame containing grade data
        output_path (str): Path to save the plot
        """
        # Calculate different types of WAM
        wam_all = self.calculate_wam(df, 'cumulative')
        wam_2000_plus = self.calculate_wam(df, 'level', 2000)  # Honours WAM
        wam_3000_plus = self.calculate_wam(df, 'level', 3000)

        # Create comparison data
        wam_types = ['Cumulative', '2000+ Level\n(Honours WAM)', '3000+ Level']
        wam_values = [wam_all['raw_wam'], wam_2000_plus['raw_wam'], wam_3000_plus['raw_wam']]

        # Create color palette with emphasis on Honours WAM
        colors = [sns.color_palette("husl")[2],
                  sns.color_palette("husl")[0],  # Highlight color for Honours WAM
                  sns.color_palette("husl")[2]]

        # Create plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(wam_types, wam_values, color=colors)

        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                     f'{height:.2f}', ha='center', va='bottom')

        # Highlight the Honours WAM bar
        plt.text(1, wam_2000_plus['raw_wam'] / 2, "DEFAULT",
                 ha='center', va='center', color='white',
                 fontweight='bold', rotation=90)

        plt.title('WAM Comparison by Course Level Inclusion')
        plt.ylabel('Weighted Average Mark')
        plt.ylim(0, 100)  # Set y-axis to mark scale
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_mark_distribution(self, df: pd.DataFrame, output_path: str = 'wam_mark_distribution.png') -> None:
        """Create a histogram of numerical marks with WAM thresholds"""
        plt.figure(figsize=(10, 6))
        valid_marks = df[df['mark'].notna()]['mark']

        # Plot histogram
        sns.histplot(data=valid_marks, bins=10, kde=True, color=sns.color_palette("husl")[2])

        # Add mean line
        plt.axvline(valid_marks.mean(), color='red', linestyle='--',
                    label=f'Mean: {valid_marks.mean():.1f}')

        # Add common WAM thresholds
        thresholds = {
            'HD (85+)': 85,
            'D (75-84)': 75,
            'C (65-74)': 65,
            'P (50-64)': 50,
            'F (<50)': 50
        }

        colors = ['darkgreen', 'green', 'orange', 'red']
        for i, (label, value) in enumerate(list(thresholds.items())[:-1]):
            plt.axvline(value, color=colors[i], linestyle='-.',
                        alpha=0.7, label=label)

        plt.title('Distribution of Marks with WAM Grade Thresholds')
        plt.xlabel('Mark')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def calculate_semester_wam(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate WAM for each semester

        Parameters:
        df (pd.DataFrame): DataFrame containing grade data

        Returns:
        pd.DataFrame: DataFrame containing semester, WAM, and units
        """
        # Extract course levels from course codes
        df_wam = df.copy()
        df_wam['course_level'] = df_wam['course_code'].str.extract(r'([0-9])')[0].astype(int) * 1000

        # Define the weighting based on course level
        level_weights = {
            1000: 1,
            2000: 2,
            3000: 3,
            4000: 4,
            5000: 4,
            6000: 4
        }

        # Apply the weights
        df_wam['weight'] = df_wam['course_level'].apply(lambda x: level_weights.get(x, 1))

        # Transform grades and marks according to WAM rules
        def map_mark_for_wam(row):
            if row['grade'] == 'UP':
                return 58
            if row['grade'] in ['P', 'C', 'D', 'HD']:
                return row['mark']
            if row['grade'] == 'F' and 45 <= row['mark'] <= 49:
                return row['mark']
            if row['grade'] == 'F' and row['mark'] < 45:
                return 44
            return row['mark']

        df_wam['wam_mark'] = df_wam.apply(map_mark_for_wam, axis=1)

        # Create a DataFrame to hold semester WAMs
        semester_wam = []

        # Get unique semesters in chronological order
        semesters = sorted(df_wam['semester'].unique())

        # Calculate WAM for each semester
        for semester in semesters:
            semester_df = df_wam[df_wam['semester'] == semester]

            # Check for UP grades and apply special handling
            up_grades_count = semester_df[semester_df['grade'] == 'UP'].shape[0]
            total_grades_count = semester_df.shape[0]
            include_up_grades = False

            # Calculate preliminary WAM without UP grades
            df_without_up = semester_df[semester_df['grade'] != 'UP']

            # Calculation: WAM = Σ (M × V × W) / Σ (V × W)
            sum_weighted_marks = (df_without_up['wam_mark'] * df_without_up['units'] * df_without_up['weight']).sum()
            sum_weighted_units = (df_without_up['units'] * df_without_up['weight']).sum()

            if sum_weighted_units > 0:
                preliminary_wam = sum_weighted_marks / sum_weighted_units
            else:
                preliminary_wam = 0

            # Check if UP grades should be included
            if up_grades_count > 0.5 * total_grades_count:  # More than 50% are UP grades
                include_up_grades = True
            elif preliminary_wam < 58:  # WAM < 58, including UP grades will help
                include_up_grades = True

            # Final WAM calculation
            if include_up_grades and up_grades_count > 0:
                sum_weighted_marks = (semester_df['wam_mark'] * semester_df['units'] * semester_df['weight']).sum()
                sum_weighted_units = (semester_df['units'] * semester_df['weight']).sum()
                final_wam = sum_weighted_marks / sum_weighted_units if sum_weighted_units > 0 else 0
            else:
                final_wam = preliminary_wam

            semester_wam.append({
                'semester': semester,
                'wam': round(final_wam, 2),
                'units': semester_df['units'].sum(),
                'up_included': include_up_grades
            })

        return pd.DataFrame(semester_wam)

    def plot_wam_trend(self, df: pd.DataFrame, output_path: str = 'wam_trend.png') -> None:
        """
        Create a line plot showing WAM trend across semesters

        Parameters:
        df (pd.DataFrame): DataFrame containing grade data
        output_path (str): Path to save the plot
        """
        # Calculate WAM for each semester
        semester_wam = self.calculate_semester_wam(df)

        # Create the figure
        plt.figure(figsize=(12, 6))

        # Plot the WAM trend line
        sns.lineplot(
            x='semester',
            y='wam',
            data=semester_wam,
            marker='o',
            markersize=10,
            linewidth=2,
            color=sns.color_palette("husl")[0]
        )

        # Calculate and plot cumulative WAM
        cumulative_points = 0
        cumulative_units = 0
        cumulative_weighted_units = 0
        cumulative_wam = []

        # Make a copy of the dataframe for cumulative calculations
        df_cum = df.copy()
        df_cum['course_level'] = df_cum['course_code'].str.extract(r'([0-9])')[0].astype(int) * 1000
        df_cum['weight'] = df_cum['course_level'].apply(
            lambda x: {1000: 1, 2000: 2, 3000: 3, 4000: 4, 5000: 4, 6000: 4}.get(x, 1))

        # Function to calculate WAM mark
        def get_wam_mark(row):
            if row['grade'] == 'UP':
                return 58
            if row['grade'] in ['P', 'C', 'D', 'HD']:
                return row['mark']
            if row['grade'] == 'F' and 45 <= row['mark'] <= 49:
                return row['mark']
            if row['grade'] == 'F' and row['mark'] < 45:
                return 44
            return row['mark']

        df_cum['wam_mark'] = df_cum.apply(get_wam_mark, axis=1)

        semesters = sorted(df_cum['semester'].unique())
        for i, semester in enumerate(semesters):
            # Get all data up to and including current semester
            current_data = df_cum[df_cum['semester'].apply(lambda x: x in semesters[:i + 1])]

            # Calculate cumulative WAM
            weighted_marks = current_data['wam_mark'] * current_data['units'] * current_data['weight']
            weighted_units = current_data['units'] * current_data['weight']

            cum_wam = weighted_marks.sum() / weighted_units.sum() if weighted_units.sum() > 0 else 0
            cumulative_wam.append(round(cum_wam, 2))

        # Plot cumulative WAM
        plt.plot(
            range(len(semester_wam)),
            cumulative_wam,
            linestyle='--',
            marker='s',
            markersize=8,
            color=sns.color_palette("husl")[1],
            label='Cumulative WAM'
        )

        # Numbers on plot points have been removed as requested

        # Set plot properties
        plt.title('WAM Trend by Semester', fontsize=16)
        plt.xlabel('Semester', fontsize=12)
        plt.ylabel('WAM', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 100)  # Setting y-axis range based on mark scale
        plt.tight_layout()
        plt.legend(['Semester WAM', 'Cumulative WAM'])

        # Save the figure
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def generate_wam_report(self, df: pd.DataFrame) -> str:
        """
        Generate a detailed WAM report as a formatted string

        Parameters:
        df (pd.DataFrame): DataFrame containing grade data

        Returns:
        str: Formatted WAM report
        """
        # Calculate WAMs
        wam_all = self.calculate_wam(df, 'cumulative')
        wam_2000_plus = self.calculate_wam(df, 'level', 2000)
        wam_3000_plus = self.calculate_wam(df, 'level', 3000)

        # Format the report
        report = []
        report.append("\nWAM (Weighted Average Mark) Report")
        report.append("=" * 50)

        report.append("\nCumulative WAM (all courses):")
        report.append(f"Raw WAM: {wam_all['raw_wam']:.2f}")
        report.append(f"Rounded WAM: {wam_all['rounded_wam']}")
        report.append(f"Total units: {wam_all['total_units']}")
        report.append(f"UP grades included: {'Yes' if wam_all['up_grades_included'] else 'No'}")

        report.append("\nWAM for 2000+ level courses:")
        report.append(f"Raw WAM: {wam_2000_plus['raw_wam']:.2f}")
        report.append(f"Rounded WAM: {wam_2000_plus['rounded_wam']}")
        report.append(f"Total units: {wam_2000_plus['total_units']}")

        report.append("\nWAM for 3000+ level courses (Honours WAM):")
        report.append(f"Raw WAM: {wam_3000_plus['raw_wam']:.2f}")
        report.append(f"Rounded WAM: {wam_3000_plus['rounded_wam']}")
        report.append(f"Total units: {wam_3000_plus['total_units']}")

        report.append("\nCourse level distribution:")
        for level, units in sorted(wam_all['course_level_counts'].items()):
            report.append(f"{level} level: {units} units")

        # Add course-level performance for WAM calculation
        report.append("\nDetailed course performance for WAM calculation:")
        report.append("-" * 80)
        report.append(
            f"{'Course':<12} {'Level':<8} {'Mark':<8} {'Grade':<6} {'Units':<8} {'Weight':<8} {'WAM Points':<12}")
        report.append("-" * 80)

        for _, row in df.sort_values('course_code').iterrows():
            level = int(row['course_code'][4]) * 1000  # Extract level from course code
            weight = 1
            if level >= 2000:
                weight = 2
            if level >= 3000:
                weight = 3
            if level >= 4000:
                weight = 4

            # Calculate WAM mark similar to map_mark_for_wam function
            wam_mark = row['mark']
            if row['grade'] == 'UP':
                wam_mark = 58
            elif row['grade'] == 'F' and row['mark'] < 45:
                wam_mark = 44

            wam_points = wam_mark * row['units'] * weight

            report.append(
                f"{row['course_code']:<12} {level:<8} {wam_mark:<8.1f} {row['grade']:<6} {row['units']:<8} {weight:<8} {wam_points:<12.1f}")

        return "\n".join(report)


def main():
    calculator = WAMCalculator()

    # Check if samplegrades.csv file exists and use it
    try:
        print("Attempting to load data from samplegrades.csv...")
        df = calculator.load_from_csv('samplegrades.csv')
        print("Successfully loaded data from samplegrades.csv.")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading CSV: {e}")
        print("Please ensure your CSV file exists and has the correct format.")
        return

    # Generate WAM report with honours calculation (2000+ level courses) as default
    try:
        # Calculate honours WAM (2000+ level courses)
        honours_wam = calculator.calculate_wam(df, 'level', 2000)

        # Print Honours WAM as the primary result
        print("\nHonours WAM Calculation (2000+ level courses)")
        print("=" * 50)
        print(f"Raw WAM: {honours_wam['raw_wam']:.2f}")
        print(f"Rounded WAM: {honours_wam['rounded_wam']}")
        print(f"Total units: {honours_wam['total_units']}")

        # Generate full WAM report for reference
        wam_report = calculator.generate_wam_report(df)
        print("\n" + "=" * 50)
        print("ADDITIONAL WAM CALCULATIONS FOR REFERENCE:")
        print(wam_report)
    except Exception as e:
        print(f"\nError calculating WAM: {e}")
        print("WAM calculation requires valid course codes with level information.")
        return

    # Generate WAM visualisations
    try:
        calculator.plot_wam_comparison(df)
        calculator.plot_mark_distribution(df)
        calculator.plot_wam_trend(df)
        print("\nGenerated visualisations:")
        print("- WAM comparison chart: wam_comparison.png")
        print("- Mark distribution with WAM thresholds: wam_mark_distribution.png")
        print("- WAM trend by semester: wam_trend.png")

        # Print semester WAM information
        semester_wam_df = calculator.calculate_semester_wam(df)
        print("\nSemester WAM Breakdown:")
        print("-" * 40)
        print(f"{'Semester':<10} {'Units':>8} {'WAM':>8}")
        print("-" * 40)
        for _, row in semester_wam_df.iterrows():
            print(f"{row['semester']:<10} {row['units']:>8} {row['wam']:>8.2f}")
    except Exception as e:
        print(f"\nError generating visualisations: {e}")


if __name__ == "__main__":
    main()