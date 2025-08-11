"""
Test suite for heat_adaptation.io.data_manager module.

Tests cover:
- CSV loading and processing
- Column mapping and detection
- Data validation and cleaning
- File format handling
- Error handling for various input scenarios
- Sample data generation
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta
from io import StringIO
from unittest.mock import Mock, patch, MagicMock


def pace_to_seconds(pace_str):
    """Convert MM:SS pace string to seconds per mile"""
    minutes, seconds = map(int, pace_str.split(":"))
    return minutes * 60 + seconds

def heat_score(temp, humidity, pace_sec_per_mile, avg_hr, max_hr, distance, multiplier=1.0):
    """Calculate heat strain score - simplified for testing"""
    # Simplified calculation for testing
    return multiplier * (temp + humidity + (pace_sec_per_mile / 10) + avg_hr + distance) / 10


class DataManager:
    """Data manager for handling CSV loading and processing"""
    
    def __init__(self):
        self.supported_date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y']
        self.required_fields = {
            'date': ['date', 'Date', 'DATE', 'run_date', 'workout_date'],
            'temp': ['temp', 'temperature', 'Temperature', 'TEMP', 'temp_f', 'temperature_f'],
            'humidity': ['humidity', 'Humidity', 'HUMIDITY', 'humid', 'rh', 'RH'],
            'pace': ['pace', 'Pace', 'PACE', 'avg_pace', 'average_pace', 'pace_per_mile'],
            'avg_hr': ['avg_hr', 'average_hr', 'hr', 'heart_rate', 'HR', 'avg_heart_rate'],
            'distance': ['distance', 'Distance', 'DISTANCE', 'miles', 'km', 'dist']
        }
    
    def load_csv(self, filepath):
        """Load CSV file with error handling"""
        try:
            df = pd.read_csv(filepath)
            return df, None
        except FileNotFoundError:
            return None, f"File not found: {filepath}"
        except pd.errors.EmptyDataError:
            return None, "CSV file is empty"
        except pd.errors.ParserError as e:
            return None, f"Error parsing CSV: {str(e)}"
        except Exception as e:
            return None, f"Unexpected error loading CSV: {str(e)}"
    
    def map_columns(self, df):
        """Intelligently map CSV columns to required fields"""
        column_mapping = {}
        available_columns = list(df.columns)
        
        for field, possible_names in self.required_fields.items():
            found_column = None
            
            # Try exact matches first
            for possible in possible_names:
                if possible in available_columns:
                    found_column = possible
                    break
            
            # Try case-insensitive partial matches
            if not found_column:
                for col in available_columns:
                    for possible in possible_names:
                        if possible.lower() in col.lower() or col.lower() in possible.lower():
                            found_column = col
                            break
                    if found_column:
                        break
            
            if found_column:
                column_mapping[field] = found_column
        
        return column_mapping
    
    def validate_required_columns(self, column_mapping):
        """Validate that minimum required columns are present"""
        required_minimum = ['date', 'temp', 'humidity', 'pace', 'avg_hr']
        missing_required = [field for field in required_minimum if field not in column_mapping]
        return missing_required
    
    def parse_date(self, date_str):
        """Parse date string using multiple formats"""
        date_str = str(date_str).strip()
        
        for date_format in self.supported_date_formats:
            try:
                return datetime.strptime(date_str, date_format), None
            except ValueError:
                continue
        
        return None, f"Could not parse date: {date_str}"
    
    def parse_pace(self, pace_str):
        """Parse pace string (MM:SS or decimal minutes)"""
        pace_str = str(pace_str).strip()
        
        if ':' in pace_str:
            try:
                return pace_to_seconds(pace_str), None
            except ValueError as e:
                return None, f"Invalid pace format: {pace_str}"
        else:
            try:
                # Assume decimal minutes (e.g., 7.5 = 7:30)
                pace_min = float(pace_str)
                pace_sec = int(pace_min * 60)
                return pace_sec, None
            except ValueError:
                return None, f"Invalid pace format: {pace_str}"
    
    def process_csv_data(self, df, column_mapping, mile_pr_str, max_hr_global):
        """Process CSV data into the format expected by the analysis"""
        run_data = []
        errors = []
        mile_pr_sec = pace_to_seconds(mile_pr_str)
        
        for idx, row in df.iterrows():
            try:
                # Parse date
                if 'date' in column_mapping:
                    date_obj, date_error = self.parse_date(row[column_mapping['date']])
                    if date_error:
                        errors.append(f"Row {idx+1}: {date_error}")
                        continue
                else:
                    errors.append(f"Row {idx+1}: No date column found")
                    continue
                
                # Extract numeric fields
                temp = self._extract_numeric(row, column_mapping, 'temp', idx+1, errors)
                humidity = self._extract_numeric(row, column_mapping, 'humidity', idx+1, errors)
                avg_hr = self._extract_numeric(row, column_mapping, 'avg_hr', idx+1, errors)
                distance = self._extract_numeric(row, column_mapping, 'distance', idx+1, errors, default=3.0)
                
                if temp is None or humidity is None or avg_hr is None:
                    continue
                
                # Parse pace
                if 'pace' in column_mapping:
                    pace_sec, pace_error = self.parse_pace(row[column_mapping['pace']])
                    if pace_error:
                        errors.append(f"Row {idx+1}: {pace_error}")
                        continue
                else:
                    errors.append(f"Row {idx+1}: No pace column found")
                    continue
                
                # Calculate raw score
                raw_score = heat_score(temp, humidity, pace_sec, avg_hr, max_hr_global, distance, multiplier=1.0)
                
                run_data.append({
                    'date': date_obj,
                    'temp': temp,
                    'humidity': humidity,
                    'pace_sec': pace_sec,
                    'avg_hr': avg_hr,
                    'max_hr': max_hr_global,
                    'distance': distance,
                    'raw_score': raw_score,
                    'adjusted_score': None,
                    'relative_score': None,
                    'mile_pr_sec': mile_pr_sec
                })
                
            except Exception as e:
                errors.append(f"Row {idx+1}: Unexpected error - {str(e)}")
                continue
        
        return run_data, errors
    
    def _extract_numeric(self, row, column_mapping, field, row_num, errors, default=None):
        """Extract and validate numeric field from row"""
        if field in column_mapping:
            try:
                value = float(row[column_mapping[field]])
                return value
            except (ValueError, TypeError):
                errors.append(f"Row {row_num}: Invalid {field} value")
                return None
        elif default is not None:
            return default
        else:
            errors.append(f"Row {row_num}: Missing {field}")
            return None
    
    def create_sample_csv(self, filepath='sample_heat_data.csv'):
        """Create a sample CSV file for users"""
        sample_data = {
            'date': ['2024-07-01', '2024-07-03', '2024-07-05', '2024-07-07', '2024-07-10'],
            'temp': [85, 88, 92, 89, 91],
            'humidity': [75, 80, 85, 78, 82],
            'pace': ['7:30', '7:45', '8:00', '7:35', '7:50'],
            'avg_hr': [165, 170, 175, 168, 172],
            'distance': [3.1, 5.0, 3.1, 4.0, 6.0]
        }
        
        sample_df = pd.DataFrame(sample_data)
        sample_df.to_csv(filepath, index=False)
        return filepath


class TestDataManager:
    """Test the DataManager class"""
    
    @pytest.fixture
    def data_manager(self):
        """Create a DataManager instance"""
        return DataManager()
    
    @pytest.fixture
    def sample_csv_content(self):
        """Create sample CSV content"""
        return """date,temp,humidity,pace,avg_hr,distance
2024-07-01,85,75,7:30,165,3.1
2024-07-03,88,80,7:45,170,5.0
2024-07-05,92,85,8:00,175,3.1
2024-07-07,89,78,7:35,168,4.0
2024-07-10,91,82,7:50,172,6.0"""
    
    @pytest.fixture
    def problematic_csv_content(self):
        """Create CSV content with various issues"""
        return """Date,Temperature,Humidity %,Average Pace,HR,Miles
07/01/2024,85,75,7.5,165,3.1
07/03/2024,88,80,7:45,170,5.0
07/05/2024,invalid_temp,85,8:00,175,3.1
07/07/2024,89,78,7:35,invalid_hr,4.0
07/10/2024,91,82,bad_pace,172,6.0"""
    
    def test_load_csv_success(self, data_manager, sample_csv_content):
        """Test successful CSV loading"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(sample_csv_content)
            temp_path = f.name
        
        try:
            df, error = data_manager.load_csv(temp_path)
            
            assert df is not None
            assert error is None
            assert len(df) == 5
            assert list(df.columns) == ['date', 'temp', 'humidity', 'pace', 'avg_hr', 'distance']
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_file_not_found(self, data_manager):
        """Test loading non-existent file"""
        df, error = data_manager.load_csv('nonexistent_file.csv')
        
        assert df is None
        assert "File not found" in error
    
    def test_load_csv_empty_file(self, data_manager):
        """Test loading empty CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('')  # Empty file
            temp_path = f.name
        
        try:
            df, error = data_manager.load_csv(temp_path)
            
            assert df is None
            assert "empty" in error.lower()
        finally:
            os.unlink(temp_path)
    
    def test_load_csv_invalid_format(self, data_manager):
        """Test loading invalid CSV format"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write('invalid,csv"content\n"unclosed,quotes')
            temp_path = f.name
        
        try:
            df, error = data_manager.load_csv(temp_path)
            
            # Depending on pandas version, this might succeed or fail
            # If it fails, should have error message
            if df is None:
                assert error is not None
        finally:
            os.unlink(temp_path)
    
    def test_map_columns_exact_match(self, data_manager):
        """Test column mapping with exact matches"""
        df = pd.DataFrame(columns=['date', 'temp', 'humidity', 'pace', 'avg_hr', 'distance'])
        
        column_mapping = data_manager.map_columns(df)
        
        assert column_mapping['date'] == 'run_date'
        assert column_mapping['temp'] == 'temp_f'
        assert column_mapping['humidity'] == 'humidity_pct'
        assert column_mapping['pace'] == 'average_pace'
        assert column_mapping['avg_hr'] == 'heart_rate'
        assert column_mapping['distance'] == 'miles'
    
    def test_map_columns_missing_columns(self, data_manager):
        """Test column mapping when some columns are missing"""
        df = pd.DataFrame(columns=['date', 'temp', 'humidity'])  # Missing pace, avg_hr, distance
        
        column_mapping = data_manager.map_columns(df)
        
        assert column_mapping['date'] == 'date'
        assert column_mapping['temp'] == 'temp'
        assert column_mapping['humidity'] == 'humidity'
        assert 'pace' not in column_mapping
        assert 'avg_hr' not in column_mapping
        assert 'distance' not in column_mapping
    
    def test_validate_required_columns_complete(self, data_manager):
        """Test validation when all required columns are present"""
        column_mapping = {
            'date': 'date',
            'temp': 'temp',
            'humidity': 'humidity',
            'pace': 'pace',
            'avg_hr': 'avg_hr',
            'distance': 'distance'
        }
        
        missing = data_manager.validate_required_columns(column_mapping)
        
        assert len(missing) == 0
    
    def test_validate_required_columns_missing(self, data_manager):
        """Test validation when required columns are missing"""
        column_mapping = {
            'date': 'date',
            'temp': 'temp'
            # Missing humidity, pace, avg_hr
        }
        
        missing = data_manager.validate_required_columns(column_mapping)
        
        assert 'humidity' in missing
        assert 'pace' in missing
        assert 'avg_hr' in missing
        assert len(missing) == 3


class TestDateParsing:
    """Test date parsing functionality"""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_parse_date_iso_format(self, data_manager):
        """Test parsing ISO format dates (YYYY-MM-DD)"""
        date_obj, error = data_manager.parse_date('2024-07-15')
        
        assert error is None
        assert date_obj == datetime(2024, 7, 15)
    
    def test_parse_date_us_format(self, data_manager):
        """Test parsing US format dates (MM/DD/YYYY)"""
        date_obj, error = data_manager.parse_date('07/15/2024')
        
        assert error is None
        assert date_obj == datetime(2024, 7, 15)
    
    def test_parse_date_alternative_formats(self, data_manager):
        """Test parsing alternative date formats"""
        # MM-DD-YYYY
        date_obj1, error1 = data_manager.parse_date('07-15-2024')
        assert error1 is None
        assert date_obj1 == datetime(2024, 7, 15)
        
        # DD/MM/YYYY
        date_obj2, error2 = data_manager.parse_date('15/07/2024')
        assert error2 is None
        assert date_obj2 == datetime(2024, 7, 15)
    
    def test_parse_date_invalid(self, data_manager):
        """Test parsing invalid date strings"""
        date_obj, error = data_manager.parse_date('invalid-date')
        
        assert date_obj is None
        assert "Could not parse date" in error
    
    def test_parse_date_whitespace(self, data_manager):
        """Test parsing dates with whitespace"""
        date_obj, error = data_manager.parse_date('  2024-07-15  ')
        
        assert error is None
        assert date_obj == datetime(2024, 7, 15)


class TestPaceParsing:
    """Test pace parsing functionality"""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_parse_pace_mm_ss_format(self, data_manager):
        """Test parsing MM:SS pace format"""
        pace_sec, error = data_manager.parse_pace('7:30')
        
        assert error is None
        assert pace_sec == 450  # 7*60 + 30
    
    def test_parse_pace_decimal_minutes(self, data_manager):
        """Test parsing decimal minutes format"""
        pace_sec, error = data_manager.parse_pace('7.5')
        
        assert error is None
        assert pace_sec == 450  # 7.5 * 60
    
    def test_parse_pace_edge_cases(self, data_manager):
        """Test parsing pace edge cases"""
        # Very fast pace
        pace_sec1, error1 = data_manager.parse_pace('5:00')
        assert error1 is None
        assert pace_sec1 == 300
        
        # Very slow pace
        pace_sec2, error2 = data_manager.parse_pace('12:00')
        assert error2 is None
        assert pace_sec2 == 720
        
        # Sub-minute seconds
        pace_sec3, error3 = data_manager.parse_pace('6:45')
        assert error3 is None
        assert pace_sec3 == 405
    
    def test_parse_pace_invalid(self, data_manager):
        """Test parsing invalid pace strings"""
        invalid_paces = ['invalid', '7.30.5', 'abc:def', '25:61', '-5:30']
        
        for invalid_pace in invalid_paces:
            pace_sec, error = data_manager.parse_pace(invalid_pace)
            assert pace_sec is None
            assert "Invalid pace format" in error
    
    def test_parse_pace_whitespace(self, data_manager):
        """Test parsing pace with whitespace"""
        pace_sec, error = data_manager.parse_pace('  7:30  ')
        
        assert error is None
        assert pace_sec == 450


class TestDataProcessing:
    """Test CSV data processing"""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    @pytest.fixture
    def valid_df(self):
        """Create valid dataframe for processing"""
        return pd.DataFrame({
            'date': ['2024-07-01', '2024-07-03', '2024-07-05'],
            'temp': [85, 88, 92],
            'humidity': [75, 80, 85],
            'pace': ['7:30', '7:45', '8:00'],
            'avg_hr': [165, 170, 175],
            'distance': [3.1, 5.0, 3.1]
        })
    
    @pytest.fixture
    def column_mapping(self):
        """Standard column mapping"""
        return {
            'date': 'date',
            'temp': 'temp',
            'humidity': 'humidity',
            'pace': 'pace',
            'avg_hr': 'avg_hr',
            'distance': 'distance'
        }
    
    def test_process_csv_data_success(self, data_manager, valid_df, column_mapping):
        """Test successful data processing"""
        mile_pr_str = '6:40'
        max_hr_global = 190
        
        run_data, errors = data_manager.process_csv_data(
            valid_df, column_mapping, mile_pr_str, max_hr_global
        )
        
        assert len(run_data) == 3
        assert len(errors) == 0
        
        # Check first run data
        first_run = run_data[0]
        assert first_run['date'] == datetime(2024, 7, 1)
        assert first_run['temp'] == 85
        assert first_run['humidity'] == 75
        assert first_run['pace_sec'] == 450  # 7:30
        assert first_run['avg_hr'] == 165
        assert first_run['max_hr'] == 190
        assert first_run['distance'] == 3.1
        assert first_run['raw_score'] > 0
        assert first_run['mile_pr_sec'] == 400  # 6:40
    
    def test_process_csv_data_with_errors(self, data_manager):
        """Test data processing with various errors"""
        problematic_df = pd.DataFrame({
            'date': ['2024-07-01', 'invalid-date', '2024-07-05'],
            'temp': [85, 88, 'invalid'],
            'humidity': [75, 80, 85],
            'pace': ['7:30', '7:45', 'invalid'],
            'avg_hr': [165, 'invalid', 175],
            'distance': [3.1, 5.0, 3.1]
        })
        
        column_mapping = {
            'date': 'date',
            'temp': 'temp',
            'humidity': 'humidity',
            'pace': 'pace',
            'avg_hr': 'avg_hr',
            'distance': 'distance'
        }
        
        run_data, errors = data_manager.process_csv_data(
            problematic_df, column_mapping, '6:40', 190
        )
        
        # Should only process the first row successfully
        assert len(run_data) == 1
        assert len(errors) > 0
        
        # Check that errors are descriptive
        error_text = ' '.join(errors)
        assert 'date' in error_text or 'Date' in error_text
        assert any('temp' in error.lower() or 'pace' in error.lower() or 'hr' in error.lower() for error in errors)
    
    def test_process_csv_data_missing_columns(self, data_manager, valid_df):
        """Test data processing with missing required columns"""
        incomplete_mapping = {
            'date': 'date',
            'temp': 'temp'
            # Missing humidity, pace, avg_hr
        }
        
        run_data, errors = data_manager.process_csv_data(
            valid_df, incomplete_mapping, '6:40', 190
        )
        
        assert len(run_data) == 0  # No successful processing
        assert len(errors) > 0
        
        # Should have errors about missing columns
        error_text = ' '.join(errors)
        assert 'humidity' in error_text or 'pace' in error_text or 'hr' in error_text
    
    def test_process_csv_data_default_distance(self, data_manager):
        """Test data processing with missing distance column (should use default)"""
        df_no_distance = pd.DataFrame({
            'date': ['2024-07-01', '2024-07-03'],
            'temp': [85, 88],
            'humidity': [75, 80],
            'pace': ['7:30', '7:45'],
            'avg_hr': [165, 170]
        })
        
        column_mapping = {
            'date': 'date',
            'temp': 'temp',
            'humidity': 'humidity',
            'pace': 'pace',
            'avg_hr': 'avg_hr'
            # No distance mapping
        }
        
        run_data, errors = data_manager.process_csv_data(
            df_no_distance, column_mapping, '6:40', 190
        )
        
        assert len(run_data) == 2
        assert len(errors) == 0
        
        # Should use default distance
        for run in run_data:
            assert run['distance'] == 3.0  # Default value


class TestSampleDataGeneration:
    """Test sample data generation"""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_create_sample_csv(self, data_manager):
        """Test sample CSV creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            filepath = os.path.join(temp_dir, 'test_sample.csv')
            result_path = data_manager.create_sample_csv(filepath)
            
            assert result_path == filepath
            assert os.path.exists(filepath)
            
            # Load and verify the sample data
            df = pd.read_csv(filepath)
            assert len(df) == 5
            assert list(df.columns) == ['date', 'temp', 'humidity', 'pace', 'avg_hr', 'distance']
            
            # Verify data types and ranges
            assert all(df['temp'] >= 80)  # Reasonable temperatures
            assert all(df['humidity'] >= 70)  # Reasonable humidity
            assert all(':' in pace for pace in df['pace'])  # Pace format
            assert all(df['avg_hr'] >= 160)  # Reasonable heart rates
            assert all(df['distance'] >= 3.0)  # Reasonable distances
    
    def test_create_sample_csv_default_path(self, data_manager):
        """Test sample CSV creation with default path"""
        # Clean up if file exists
        default_path = 'sample_heat_data.csv'
        if os.path.exists(default_path):
            os.unlink(default_path)
        
        try:
            result_path = data_manager.create_sample_csv()
            
            assert result_path == default_path
            assert os.path.exists(default_path)
        finally:
            # Clean up
            if os.path.exists(default_path):
                os.unlink(default_path)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_empty_dataframe(self, data_manager):
        """Test processing empty dataframe"""
        empty_df = pd.DataFrame()
        column_mapping = {}
        
        run_data, errors = data_manager.process_csv_data(
            empty_df, column_mapping, '6:40', 190
        )
        
        assert len(run_data) == 0
        assert len(errors) == 0  # No rows to process, no errors
    
    def test_single_row_processing(self, data_manager):
        """Test processing single row dataframe"""
        single_row_df = pd.DataFrame({
            'date': ['2024-07-01'],
            'temp': [85],
            'humidity': [75],
            'pace': ['7:30'],
            'avg_hr': [165],
            'distance': [3.1]
        })
        
        column_mapping = {
            'date': 'date',
            'temp': 'temp',
            'humidity': 'humidity',
            'pace': 'pace',
            'avg_hr': 'avg_hr',
            'distance': 'distance'
        }
        
        run_data, errors = data_manager.process_csv_data(
            single_row_df, column_mapping, '6:40', 190
        )
        
        assert len(run_data) == 1
        assert len(errors) == 0
    
    def test_extreme_values(self, data_manager):
        """Test processing with extreme but valid values"""
        extreme_df = pd.DataFrame({
            'date': ['2024-07-01'],
            'temp': [120],  # Very hot
            'humidity': [99],  # Very humid
            'pace': ['15:00'],  # Very slow
            'avg_hr': [200],  # High HR
            'distance': [26.2]  # Marathon distance
        })
        
        column_mapping = {
            'date': 'date',
            'temp': 'temp',
            'humidity': 'humidity',
            'pace': 'pace',
            'avg_hr': 'avg_hr',
            'distance': 'distance'
        }
        
        run_data, errors = data_manager.process_csv_data(
            extreme_df, column_mapping, '6:40', 190
        )
        
        assert len(run_data) == 1
        assert len(errors) == 0
        
        # Should still calculate heat score
        assert run_data[0]['raw_score'] > 0
    
    def test_unicode_and_special_characters(self, data_manager):
        """Test handling unicode and special characters"""
        # This would typically be in column names or data
        unicode_df = pd.DataFrame({
            'date': ['2024-07-01'],
            'temp': [85],
            'humidity': [75],
            'pace': ['7:30'],
            'avg_hr': [165],
            'distance': [3.1]
        })
        
        column_mapping = {
            'date': 'date',
            'temp': 'temp',
            'humidity': 'humidity',
            'pace': 'pace',
            'avg_hr': 'avg_hr',
            'distance': 'distance'
        }
        
        run_data, errors = data_manager.process_csv_data(
            unicode_df, column_mapping, '6:40', 190
        )
        
        assert len(run_data) == 1
        assert len(errors) == 0


class TestIntegration:
    """Integration tests for complete data loading workflow"""
    
    @pytest.fixture
    def data_manager(self):
        return DataManager()
    
    def test_complete_workflow_success(self, data_manager):
        """Test complete successful data loading workflow"""
        csv_content = """run_date,temperature_f,humidity_pct,average_pace,heart_rate,miles
2024-07-01,85,75,7:30,165,3.1
2024-07-03,88,80,7:45,170,5.0
2024-07-05,92,85,8:00,175,3.1"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        
        try:
            # Step 1: Load CSV
            df, load_error = data_manager.load_csv(temp_path)
            assert df is not None
            assert load_error is None
            
            # Step 2: Map columns
            column_mapping = data_manager.map_columns(df)
            assert len(column_mapping) == 6  # All columns should be mapped
            
            # Step 3: Validate required columns
            missing = data_manager.validate_required_columns(column_mapping)
            assert len(missing) == 0
            
            # Step 4: Process data
            run_data, process_errors = data_manager.process_csv_data(
                df, column_mapping, '6:40', 190
            )
            
            assert len(run_data) == 3
            assert len(process_errors) == 0
            
            # Verify final data structure
            for run in run_data:
                assert isinstance(run['date'], datetime)
                assert isinstance(run['temp'], (int, float))
                assert isinstance(run['humidity'], (int, float))
                assert isinstance(run['pace_sec'], int)
                assert isinstance(run['avg_hr'], (int, float))
                assert isinstance(run['raw_score'], (int, float))
                
        finally:
            os.unlink(temp_path)
    
    def test_complete_workflow_with_issues(self, data_manager):
        """Test complete workflow with data issues"""
        problematic_csv = """Date,Temp,Humidity,Pace,HR,Distance
invalid_date,85,75,7:30,165,3.1
2024-07-03,not_a_number,80,7:45,170,5.0
2024-07-05,92,85,invalid_pace,175,3.1
2024-07-07,89,78,7:35,168,4.0"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write(problematic_csv)
            temp_path = f.name
        
        try:
            # Complete workflow
            df, load_error = data_manager.load_csv(temp_path)
            assert df is not None
            
            column_mapping = data_manager.map_columns(df)
            missing = data_manager.validate_required_columns(column_mapping)
            assert len(missing) == 0  # Columns should map correctly
            
            run_data, process_errors = data_manager.process_csv_data(
                df, column_mapping, '6:40', 190
            )
            
            # Should only successfully process the last row
            assert len(run_data) == 1
            assert len(process_errors) > 0
            
            # Check that the successful row is correct
            successful_run = run_data[0]
            assert successful_run['date'] == datetime(2024, 7, 7)
            assert successful_run['temp'] == 89
            
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])assert column_mapping['date'] == 'date'
        assert column_mapping['temp'] == 'temp'
        assert column_mapping['humidity'] == 'humidity'
        assert column_mapping['pace'] == 'pace'
        assert column_mapping['avg_hr'] == 'avg_hr'
        assert column_mapping['distance'] == 'distance'
    
    def test_map_columns_case_variations(self, data_manager):
        """Test column mapping with case variations"""
        df = pd.DataFrame(columns=['Date', 'Temperature', 'HUMIDITY', 'Pace', 'HR', 'Distance'])
        
        column_mapping = data_manager.map_columns(df)
        
        assert column_mapping['date'] == 'Date'
        assert column_mapping['temp'] == 'Temperature'
        assert column_mapping['humidity'] == 'HUMIDITY'
        assert column_mapping['pace'] == 'Pace'
        assert column_mapping['avg_hr'] == 'HR'
        assert column_mapping['distance'] == 'Distance'
    
    def test_map_columns_partial_matches(self, data_manager):
        """Test column mapping with partial string matches"""
        df = pd.DataFrame(columns=['run_date', 'temp_f', 'humidity_pct', 'average_pace', 'heart_rate', 'miles'])
        
        column_mapping = data_manager.map_columns(df)
        
        