import os
import sys
import pytest
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wam import WAMCalculator


class TestWAMCalculator:
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing"""
        data = {
            'semester': ['2022-S1', '2022-S1', '2022-S2', '2022-S2'],
            'course_code': ['COMP1010', 'INFO1010', 'COMP2010', 'INFO3010'],
            'grade': ['HD', 'D', 'C', 'P'],
            'mark': [90, 82, 75, 65],
            'units': [10, 10, 10, 10],
            'class_title': ['Intro to Programming', 'Intro to Info Systems',
                            'Data Structures', 'Advanced Databases']
        }
        return pd.DataFrame(data)

    @pytest.fixture
    def calculator(self):
        """Create a WAMCalculator instance"""
        return WAMCalculator()

    def test_init(self, calculator):
        """Test that the calculator initializes correctly"""
        assert calculator is not None

    def test_calculate_wam_cumulative(self, calculator, sample_df):
        """Test cumulative WAM calculation"""
        result = calculator.calculate_wam(sample_df, 'cumulative')

        # Verify the basic structure of the result
        assert 'raw_wam' in result
        assert 'rounded_wam' in result
        assert 'honours_class' in result
        assert 'total_units' in result

        # Sample data has 4 courses of 10 units each = 40 units
        assert result['total_units'] == 40

        # Calculation will be based on actual marks with level weighting
        # COMP1010: 90 * 10 * 1 = 900
        # INFO1010: 82 * 10 * 1 = 820
        # COMP2010: 75 * 10 * 2 = 1500
        # INFO3010: 65 * 10 * 3 = 1950
        # Total weighted mark = 5170
        # Total weighted units = 10*1 + 10*1 + 10*2 + 10*3 = 70
        # WAM = 5170 / 70 = 73.86

        assert abs(result['raw_wam'] - 73.86) < 0.1
        assert result['rounded_wam'] == 74  # Rounded WAM

    def test_calculate_wam_level(self, calculator, sample_df):
        """Test level-specific WAM calculation"""
        # Test WAM for 2000+ level courses
        result_2000 = calculator.calculate_wam(sample_df, 'level', 2000)

        # Should only include COMP2010 and INFO3010
        # COMP2010: 75 * 10 * 2 = 1500
        # INFO3010: 65 * 10 * 3 = 1950
        # Total = 3450, Units = 10*2 + 10*3 = 50
        # WAM = 3450 / 50 = 69

        assert result_2000['included_courses'] == 2
        assert abs(result_2000['raw_wam'] - 69) < 0.1

        # Test WAM for 3000+ level courses
        result_3000 = calculator.calculate_wam(sample_df, 'level', 3000)

        # Should only include INFO3010
        # INFO3010: 65 * 10 * 3 = 1950
        # Total = 1950, Units = 10*3 = 30
        # WAM = 1950 / 30 = 65

        assert result_3000['included_courses'] == 1
        assert abs(result_3000['raw_wam'] - 65) < 0.1

    def test_calculate_semester_wam(self, calculator, sample_df):
        """Test semester WAM calculation"""
        result = calculator.calculate_semester_wam(sample_df)

        # Should have 2 semesters
        assert len(result) == 2

        # First semester:
        # COMP1010: 90 * 10 * 1 = 900
        # INFO1010: 82 * 10 * 1 = 820
        # Total = 1720, Units = 10*1 + 10*1 = 20
        # WAM = 1720 / 20 = 86

        # Second semester:
        # COMP2010: 75 * 10 * 2 = 1500
        # INFO3010: 65 * 10 * 3 = 1950
        # Total = 3450, Units = 10*2 + 10*3 = 50
        # WAM = 3450 / 50 = 69

        first_sem = result[result['semester'] == '2022-S1'].iloc[0]
        second_sem = result[result['semester'] == '2022-S2'].iloc[0]

        assert abs(first_sem['wam'] - 86) < 0.1
        assert abs(second_sem['wam'] - 69) < 0.1

    def test_load_from_csv(self, calculator, tmp_path):
        """Test loading data from a CSV file"""
        # Create a temporary CSV file
        csv_path = tmp_path / "test_grades.csv"
        sample_data = {
            'semester': ['2022-S1', '2022-S2'],
            'course_code': ['COMP1010', 'INFO3010'],
            'grade': ['HD', 'P'],
            'mark': [90, 65],
            'units': [10, 10],
            'class_title': ['Intro to Programming', 'Advanced Databases']
        }
        pd.DataFrame(sample_data).to_csv(csv_path, index=False)

        # Load the data
        df = calculator.load_from_csv(str(csv_path))

        # Check that it loaded correctly
        assert len(df) == 2
        assert list(df.columns) == ['semester', 'course_code', 'grade', 'mark', 'units', 'class_title']
        assert df['grade'].tolist() == ['HD', 'P']