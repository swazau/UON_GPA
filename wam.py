import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import os


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

        # Determine honours class based on WAM
        honours_class = ""
        if final_wam >= 77:
            honours_class = "Class I"
        elif final_wam >= 72:
            honours_class = "Class II Division 1"
        elif final_wam >= 67:
            honours_class = "Class II Division 2"
        else:
            honours_class = "Ungraded"

        # Prepare result dictionary
        result = {
            'calculation_type': calculation_type,
            'included_courses': len(df_wam),
            'total_units': df_wam['units'].sum(),
            'up_grades_included': include_up_grades,
            'raw_wam': final_wam,
            'rounded_wam': rounded_wam,
            'honours_class': honours_class,
            'course_level_counts': df_wam.groupby('course_level')['units'].sum().to_dict()
        }

        return result

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

        # Add honors class thresholds
        thresholds = {
            'Class I (77+)': 77,
            'Class II Div 1 (72-76)': 72,
            'Class II Div 2 (67-71)': 67,
            'Ungraded (<67)': 67
        }

        colors = ['darkgreen', 'green', 'orange', 'gray']
        for i, (label, value) in enumerate(list(thresholds.items())[:-1]):
            plt.axvline(value, color=colors[i], linestyle='-.',
                        alpha=0.7, label=label)

        plt.title('Distribution of Marks with Honours Class Thresholds')
        plt.xlabel('Mark')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()

    def plot_honours_thresholds(self, df: pd.DataFrame, output_path: str = 'honours_thresholds.png') -> None:

        # Calculate Honours WAM (2000+ level courses)
        honours_wam = self.calculate_wam(df, 'level', 2000)

        plt.figure(figsize=(14, 8))

        thresholds = [
            {'min': 77, 'max': 100, 'label': 'Class I', 'color': '#4CAF50'},  # Green
            {'min': 72, 'max': 77, 'label': 'Class II Division 1', 'color': '#8BC34A'},  # Light Green
            {'min': 67, 'max': 72, 'label': 'Class II Division 2', 'color': '#FFC107'},  # Amber
            {'min': 60, 'max': 67, 'label': 'Ungraded', 'color': '#E0E0E0'}  # Light Grey
        ]

        ax = plt.gca()
        ax.add_patch(plt.Rectangle((60, 0.1), 40, 0.8, color='#F5F5F5', alpha=0.5, zorder=0))
        y_pos = 0.5

        for threshold in thresholds:
            plt.barh(y_pos, threshold['max'] - threshold['min'], left=threshold['min'],
                     height=0.4, color=threshold['color'], alpha=0.9,
                     edgecolor='white', linewidth=1, zorder=2)

        for threshold in thresholds:
            mid_point = threshold['min'] + (threshold['max'] - threshold['min']) / 2
            # Only add label if range is wide enough
            if threshold['max'] - threshold['min'] > 5:
                plt.text(mid_point, y_pos, threshold['label'],
                         ha='center', va='center', color='white', fontweight='bold',
                         fontsize=14, zorder=3)

        line_color = '#E53935'  # Bright red
        plt.axvline(x=honours_wam['raw_wam'], color=line_color, linestyle='-', linewidth=3, zorder=4)

        text_box = plt.text(honours_wam['raw_wam'], 1.3,
                            f"Your WAM: {honours_wam['raw_wam']:.2f}",
                            ha='center', va='center', color='black', fontsize=16, fontweight='bold',
                            bbox=dict(facecolor='white', edgecolor=line_color, alpha=0.95,
                                      boxstyle='round,pad=0.6', linewidth=2), zorder=5)

        arrow_props = dict(arrowstyle='->', color=line_color, linewidth=3, shrinkA=0, shrinkB=0)
        plt.annotate('', xy=(honours_wam['raw_wam'], 0.7), xytext=(honours_wam['raw_wam'], 0.85),
                     arrowprops=arrow_props, zorder=5)

        # Set plot properties
        plt.xlim(60, 100)  # Focus on the relevant range
        plt.ylim(0, 2)  # Keep space for the labels

        # Remove axes but keep bottom axis ticks for reference
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Customise x-axis with cleaner tick marks
        plt.xticks([67, 72, 77, 85, 90, 95, 100], fontsize=12)
        ax.tick_params(axis='x', which='both', length=5, width=1, direction='out', pad=10)

        # Add threshold labels below the axis with more spacing
        threshold_labels = [
            {'pos': 67, 'text': 'Class II Div 2\nThreshold'},
            {'pos': 72, 'text': 'Class II Div 1\nThreshold'},
            {'pos': 77, 'text': 'Class I\nThreshold'}
        ]

        for label in threshold_labels:
            plt.annotate(label['text'], xy=(label['pos'], -0.1), xycoords=('data', 'axes fraction'),
                         ha='center', va='top', fontsize=11,
                         bbox=dict(boxstyle='round,pad=0.4', fc='#F5F5F5', ec='#CCCCCC', alpha=0.8))

        # Add title with improved styling
        plt.title('Honours Classification', fontsize=22, fontweight='bold', pad=20)

        # Save the figure with tight layout
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory


def main():
    calculator = WAMCalculator()

    # Check if samplegrades.csv file exists and use it
    try:
        df = calculator.load_from_csv('samplegrades.csv')
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading CSV: {e}")
        return

    # Calculate honours WAM (2000+ level courses)
    honours_wam = calculator.calculate_wam(df, 'level', 2000)

    # Print only the Honours WAM result with minimal output
    print("\nHonours WAM Calculation (2000+ level courses)")
    print("=" * 50)
    print(f"Raw WAM: {honours_wam['raw_wam']:.2f}")
    print(f"Rounded WAM: {honours_wam['rounded_wam']}")
    print(f"Honours Class: {honours_wam['honours_class']}")
    print(f"Total units: {honours_wam['total_units']}")

    wam_dir = ensure_dir("WAM")

    # Generate WAM visualisations
    try:
        calculator.plot_wam_comparison(df, os.path.join(wam_dir, 'wam_comparison.png'))
        calculator.plot_mark_distribution(df, os.path.join(wam_dir, 'wam_mark_distribution.png'))
        calculator.plot_wam_trend(df, os.path.join(wam_dir, 'wam_trend.png'))
        calculator.plot_honours_thresholds(df, os.path.join(wam_dir, 'honours_thresholds.png'))

        print("\nGenerated visualisations in WAM directory")
    except Exception as e:
        print(f"\nError generating visualizations: {e}")


if __name__ == "__main__":
    main()