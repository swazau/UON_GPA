import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import os
import argparse
import plotly.express as px
import plotly.graph_objects as go

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

    def plot_wam_trend(self, df: pd.DataFrame) -> 'plotly.graph_objs.Figure':
        """
        Create an interactive line plot showing WAM trend across semesters.
        
        Args:
            df: DataFrame with columns 'semester', 'course_code', 'mark', 'grade', 'units'.
        
        Returns:
            plotly.graph_objs.Figure: Plotly figure object for the WAM trend.
        """
        # Assume calculate_semester_wam returns a DataFrame with 'semester', 'wam', 'units'
        semester_wam = self.calculate_semester_wam(df)
        fig = go.Figure()

        # Plot Semester WAM
        fig.add_trace(go.Scatter(
            x=semester_wam['semester'],
            y=semester_wam['wam'],
            mode='lines+markers',
            name='Semester WAM',
            marker=dict(size=10, color=px.colors.qualitative.Set2[0]),
            line=dict(color=px.colors.qualitative.Set2[0])
        ))

        # Compute Cumulative WAM
        df_wam = df.copy()
        df_wam['course_level'] = df_wam['course_code'].str.extract(r'([0-9])')[0].astype(int) * 1000
        level_weights = {1000: 1, 2000: 2, 3000: 3, 4000: 4, 5000: 4, 6000: 4}
        df_wam['weight'] = df_wam['course_level'].map(level_weights).fillna(1)
        
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
        df_wam = df_wam.sort_values('semester')

        cumulative_points = 0
        cumulative_weighted_units = 0
        cumulative_wam = []

        for semester in semester_wam['semester']:
            semester_data = df_wam[df_wam['semester'] == semester]
            semester_points = (semester_data['wam_mark'] * semester_data['units'] * semester_data['weight']).sum()
            semester_weighted_units = (semester_data['units'] * semester_data['weight']).sum()
            cumulative_points += semester_points
            cumulative_weighted_units += semester_weighted_units
            cumulative_wam.append(cumulative_points / cumulative_weighted_units if cumulative_weighted_units > 0 else 0)

        # Plot Cumulative WAM
        fig.add_trace(go.Scatter(
            x=semester_wam['semester'],
            y=cumulative_wam,
            mode='lines+markers',
            name='Cumulative WAM',
            marker=dict(size=8, symbol='square', color=px.colors.qualitative.Set2[1]),
            line=dict(dash='dash', color=px.colors.qualitative.Set2[1])
        ))

        # Update layout
        fig.update_layout(
            title='WAM Trend by Semester',
            xaxis_title='Semester',
            yaxis_title='WAM',
            yaxis_range=[0, 100],
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig

    def plot_wam_comparison(self, df: pd.DataFrame) -> 'plotly.graph_objs.Figure':
        """
        Create an interactive bar chart comparing different WAM calculations.
        
        Args:
            df: DataFrame with course data.
        
        Returns:
            plotly.graph_objs.Figure: Plotly figure object for WAM comparison.
        """
        # Assume calculate_wam returns a dict with 'raw_wam' key
        wam_all = self.calculate_wam(df, 'cumulative')
        wam_2000_plus = self.calculate_wam(df, 'level', 2000)
        wam_3000_plus = self.calculate_wam(df, 'level', 3000)

        df_comparison = pd.DataFrame({
            'Type': ['Cumulative', '2000+ Level (Honours WAM)', '3000+ Level'],
            'WAM': [wam_all['raw_wam'], wam_2000_plus['raw_wam'], wam_3000_plus['raw_wam']]
        })

        fig = px.bar(df_comparison, x='Type', y='WAM', color='Type',
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     title='WAM Comparison by Course Level Inclusion')
        fig.update_layout(yaxis_range=[0, 100], showlegend=False)
        return fig

    def plot_mark_distribution(self, df: pd.DataFrame) -> 'plotly.graph_objs.Figure':
        """
        Create an interactive histogram of numerical marks with thresholds.
        
        Args:
            df: DataFrame with 'mark' column.
        
        Returns:
            plotly.graph_objs.Figure: Plotly figure object for mark distribution.
        """
        valid_marks = df[df['mark'].notna()]['mark']
        fig = px.histogram(valid_marks, nbins=10, title='Distribution of Marks with Honours Class Thresholds')
        fig.update_traces(marker_color=px.colors.qualitative.Set2[2])

        mean_mark = valid_marks.mean()
        fig.add_vline(x=mean_mark, line_dash='dash', line_color='red',
                      annotation_text=f'Mean: {mean_mark:.1f}', annotation_position='top right')

        # Add honours thresholds
        thresholds = [67, 72, 77]
        for threshold in thresholds:
            fig.add_vline(x=threshold, line_dash='dot', line_color='gray')

        fig.update_layout(xaxis_title='Mark', yaxis_title='Count')
        return fig

    def plot_honours_thresholds(self, df: pd.DataFrame) -> 'plotly.graph_objs.Figure':
        """
        Create an interactive chart showing honours classification thresholds.
        
        Args:
            df: DataFrame with course data.
        
        Returns:
            plotly.graph_objs.Figure: Plotly figure object for honours thresholds.
        """
        honours_wam = self.calculate_wam(df, 'level', 2000)
        thresholds = [
            {'min': 77, 'max': 100, 'label': 'Class I', 'color': '#4CAF50'},
            {'min': 72, 'max': 77, 'label': 'Class II Division 1', 'color': '#8BC34A'},
            {'min': 67, 'max': 72, 'label': 'Class II Division 2', 'color': '#FFC107'},
            {'min': 60, 'max': 67, 'label': 'Ungraded', 'color': '#E0E0E0'}
        ]

        fig = go.Figure()
        for threshold in thresholds:
            fig.add_trace(go.Bar(
                y=[threshold['label']],
                x=[threshold['max'] - threshold['min']],
                base=threshold['min'],
                orientation='h',
                marker_color=threshold['color'],
                name=threshold['label']
            ))

        fig.add_vline(x=honours_wam['raw_wam'], line_color='red', line_width=3,
                      annotation_text=f"Your WAM: {honours_wam['raw_wam']:.2f}",
                      annotation_position='top left')

        fig.update_layout(
            title='Honours Classification',
            xaxis_title='WAM',
            yaxis_title='Honours Class',
            barmode='stack',
            showlegend=False,
            xaxis_range=[60, 100]
        )
        return fig

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    return directory


def main():
    try:
        import transcript_processor
    except ImportError:
        print("Warning: transcript_processor module not found. PDF import functionality will be disabled.")
        transcript_processor = None

    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Calculate WAM from grade data')
    parser.add_argument('file', nargs='?', help='CSV or PDF file containing grade data')
    parser.add_argument('--pdf', action='store_true', help='Process input as PDF transcript')
    args = parser.parse_args()

    calculator = WAMCalculator()
    csv_path = 'samplegrades.csv'  # Default

    # Handle user-specified file
    if args.file:
        if args.pdf:
            # Process PDF transcript if --pdf flag is set
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

    # Load data and continue with existing WAM calculation
    try:
        df = calculator.load_from_csv(csv_path)
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