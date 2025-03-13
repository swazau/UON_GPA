import os
import sys
import pytest
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gpa import GPAVisualiser


class TestGPAVisualiser:
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        data = {
            'semester': ['2022-S1', '2022-S1', '2022-S2', '2022-S2'],
            'course_code': ['COMP1000', 'INFO1000', 'COMP2000', 'INFO2000'],
            'grade': ['HD', 'D', 'C', 'P'],
            'mark': [90, 82, 75, 65],
            'units': [10, 10, 10, 10],
            'class_title': ['Intro to Programming', 'Intro to Info Systems',
                            'Data Structures', 'Database Systems']
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def visualiser(self):
        """Create a GPAVisualiser instance"""
        return GPAVisualiser()

    def test_init(self, visualiser):
        """Test that the visualiser initializes correctly"""
        assert visualiser is not None
        assert visualiser.grade_points == {
            'HD': 7,
            'D': 6,
            'C': 5,
            'P': 4,
            'F': 0
        }

    def test_calculate_university_gpa(self, visualiser, sample_df):
        """Test GPA calculation"""
        result = visualiser.calculate_university_gpa(sample_df)

        # Expected results based on the sample data
        # HD (10 units) = 7 points * 10 = 70
        # D (10 units) = 6 points * 10 = 60
        # C (10 units) = 5 points * 10 = 50
        # P (10 units) = 4 points * 10 = 40
        # Total: 220 points / 40 units = 5.5 GPA

        assert result['total_points'] == 220
        assert result['total_units'] == 40
        assert result['gpa'] == 5.5

        # Check grade distribution
        assert result['grade_units']['HD'] == 10
        assert result['grade_units']['D'] == 10
        assert result['grade_units']['C'] == 10
        assert result['grade_units']['P'] == 10

    def test_calculate_semester_gpa(self, visualiser, sample_df):
        """Test semester GPA calculation"""
        result = visualiser.calculate_semester_gpa(sample_df)

        # Should have 2 semesters
        assert len(result) == 2

        # First semester: HD (7) and D (6) = 6.5 GPA
        # Second semester: C (5) and P (4) = 4.5 GPA
        first_sem = result[result['semester'] == '2022-S1'].iloc[0]
        second_sem = result[result['semester'] == '2022-S2'].iloc[0]

        assert first_sem['gpa'] == 6.5
        assert second_sem['gpa'] == 4.5
        assert first_sem['units'] == 20
        assert second_sem['units'] == 20

    def test_load_from_csv(self, visualiser, tmp_path):
        """Test loading data from a CSV file"""
        # Create a temporary CSV file
        csv_path = tmp_path / "test_grades.csv"
        sample_data = {
            'semester': ['2022-S1', '2022-S2'],
            'course_code': ['COMP1000', 'INFO2000'],
            'grade': ['HD', 'P'],
            'mark': [90, 65],
            'units': [10, 10],
            'class_title': ['Intro to Programming', 'Database Systems']
        }
        pd.DataFrame(sample_data).to_csv(csv_path, index=False)

        # Load the data
        df = visualiser.load_from_csv(str(csv_path))

        # Check that it loaded correctly
        assert len(df) == 2
        assert list(df.columns) == ['semester', 'course_code', 'grade', 'mark', 'units', 'class_title']
        assert df['grade'].tolist() == ['HD', 'P']