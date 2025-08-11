"""
Data Manager for Heat Adaptation Analysis

Handles CSV upload, column mapping, data processing, and manual data entry
for heat adaptation analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple


def pace_to_seconds(pace_str: str) -> int:
    """Convert pace string (MM:SS) to seconds per mile"""
    minutes, seconds = map(int, pace_str.split(":"))
    return minutes * 60 + seconds


def load_data_from_csv(filepath: Optional[str] = None) -> Optional[pd.DataFrame]:
    """Load running data from CSV file with automatic column detection"""
    
    # Try different methods to load the file
    df = None
    
    # Method 1: Direct file path
    if filepath:
        try:
            df = pd.read_csv(filepath)
            print(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        except Exception as e:
            print(f"❌ Error loading CSV: {str(e)}")
    
    # Method 2: Try to detect if running in Google Colab
    if df is None:
        try:
            import google.colab
            from google.colab import files
            print("📱 Detected Google Colab environment")
            print("📁 Please use the file upload widget that will appear...")
            
            uploaded = files.upload()
            if uploaded:
                filename = list(uploaded.keys())[0]
                df = pd.read_csv(filename)
                print(f"✅ Successfully loaded uploaded CSV: {filename}")
                print(f"📊 {len(df)} rows and {len(df.columns)} columns")
        except ImportError:
            pass  # Not in Colab
        except Exception as e:
            print(f"❌ Colab upload error: {str(e)}")
    
    # Method 3: Try to detect if running in Jupyter
    if df is None:
        try:
            from IPython.display import display, HTML
            import ipywidgets as widgets
            print("📱 Detected Jupyter environment")
            print("💡 Try using: df = pd.read_csv('your_file.csv') manually")
            print("💡 Or upload file to same directory as notebook")
        except ImportError:
            pass
    
    if df is not None:
        print(f"📊 Columns found: {list(df.columns)}")
        
        # Display first few rows for verification
        print(f"\n📋 First 3 rows preview:")
        print(df.head(3).to_string(index=False))
        
        return df
    
    return None


def get_csv_upload_instructions():
    """Provide platform-specific upload instructions"""
    print(f"\n📁 CSV UPLOAD INSTRUCTIONS:")
    print(f"=" * 40)
    
    print(f"🖥️  LOCAL COMPUTER:")
    print(f"   • Save CSV file to your computer")
    print(f"   • Enter full file path when prompted")
    print(f"   • Example: C:\\Users\\Name\\Desktop\\data.csv")
    print(f"   • Or just filename if in same folder: data.csv")
    
    print(f"\n☁️  GOOGLE COLAB:")
    print(f"   • Choose option 2 (CSV upload)")
    print(f"   • A file picker will appear automatically")
    print(f"   • Click and select your CSV file")
    
    print(f"\n📓 JUPYTER NOTEBOOK:")
    print(f"   • Upload CSV to same folder as notebook")
    print(f"   • Or use full file path")
    
    print(f"\n🌐 ONLINE PLATFORMS:")
    print(f"   • May need to upload file first to platform")
    print(f"   • Then use filename or path as provided by platform")
    
    print(f"\n💡 ALTERNATIVE: Copy-paste data manually")
    print(f"   • Choose option 1 (manual entry)")
    print(f"   • Enter each run individually")


def map_csv_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Intelligently map CSV columns to required fields"""
    required_fields = {
        'date': ['date', 'Date', 'DATE', 'run_date', 'workout_date'],
        'temp': ['temp', 'temperature', 'Temperature', 'TEMP', 'temp_f', 'temperature_f'],
        'humidity': ['humidity', 'Humidity', 'HUMIDITY', 'humid', 'rh', 'RH'],
        'pace': ['pace', 'Pace', 'PACE', 'avg_pace', 'average_pace', 'pace_per_mile'],
        'avg_hr': ['avg_hr', 'average_hr', 'hr', 'heart_rate', 'HR', 'avg_heart_rate'],
        'distance': ['distance', 'Distance', 'DISTANCE', 'miles', 'km', 'dist']
    }
    
    column_mapping = {}
    available_columns = list(df.columns)
    
    print(f"\n🔍 SMART COLUMN MAPPING")
    print(f"Available columns: {available_columns}")
    
    for field, possible_names in required_fields.items():
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
            print(f"   ✅ {field}: {found_column}")
        else:
            print(f"   ❌ {field}: NOT FOUND")
    
    return column_mapping


def process_csv_data(df: pd.DataFrame, column_mapping: Dict[str, str], 
                    mile_pr_str: str, max_hr_global: float) -> List[Dict]:
    """Process CSV data into the format expected by the analysis"""
    from ..core.calculations import heat_score
    
    run_data = []
    mile_pr_sec = pace_to_seconds(mile_pr_str)
    
    print(f"\n🔄 PROCESSING CSV DATA...")
    
    for idx, row in df.iterrows():
        try:
            # Extract and convert date
            if 'date' in column_mapping:
                date_str = str(row[column_mapping['date']])
                # Handle multiple date formats
                for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y']:
                    try:
                        date_obj = datetime.strptime(date_str, date_format)
                        break
                    except ValueError:
                        continue
                else:
                    print(f"   ⚠️ Warning: Could not parse date '{date_str}' in row {idx+1}")
                    continue
            else:
                print(f"   ❌ No date column found. Skipping row {idx+1}")
                continue
            
            # Extract temperature
            temp = float(row[column_mapping['temp']]) if 'temp' in column_mapping else None
            
            # Extract humidity
            humidity = float(row[column_mapping['humidity']]) if 'humidity' in column_mapping else None
            
            # Extract and convert pace
            if 'pace' in column_mapping:
                pace_str = str(row[column_mapping['pace']])
                # Handle different pace formats
                if ':' in pace_str:
                    pace_sec = pace_to_seconds(pace_str)
                else:
                    # Assume decimal minutes (e.g., 7.5 = 7:30)
                    pace_min = float(pace_str)
                    pace_sec = int(pace_min * 60)
            else:
                print(f"   ❌ No pace column found. Skipping row {idx+1}")
                continue
            
            # Extract heart rate
            avg_hr = float(row[column_mapping['avg_hr']]) if 'avg_hr' in column_mapping else None
            
            # Extract distance
            distance = float(row[column_mapping['distance']]) if 'distance' in column_mapping else 3.0  # Default 5K
            
            # Validate required fields
            if temp is None or humidity is None or avg_hr is None:
                missing = [f for f, col in [('temp', temp), ('humidity', humidity), ('avg_hr', avg_hr)] if col is None]
                print(f"   ⚠️ Row {idx+1}: Missing {', '.join(missing)}. Skipping.")
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
            print(f"   ❌ Error processing row {idx+1}: {str(e)}")
            continue
    
    print(f"   ✅ Successfully processed {len(run_data)} out of {len(df)} rows")
    return run_data


def create_sample_csv():
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
    sample_df.to_csv('sample_heat_data.csv', index=False)
    print(f"📄 Sample CSV created: 'sample_heat_data.csv'")
    print(f"📋 Use this as a template for your own data!")


def collect_manual_data(num_entries: int, mile_pr_str: str, max_hr_global: float) -> List[Dict]:
    """Collect run data through manual entry"""
    from ..core.calculations import heat_score
    
    run_data = []
    mile_pr_sec = pace_to_seconds(mile_pr_str)

    for i in range(num_entries):
        print(f"\n--- Run {i+1}/{num_entries} ---")
        date_str = input("Date of run (YYYY-MM-DD): ")
        temp = float(input("Temperature (°F): "))
        humidity = float(input("Humidity (%): "))
        pace_str = input("Pace (MM:SS per mile): ")
        avg_hr = float(input("Average HR: "))
        distance = float(input("Distance (miles): "))

        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        pace_sec = pace_to_seconds(pace_str)

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

    return run_data


def get_data_input() -> Tuple[List[Dict], str, float]:
    """Main function to handle data input (CSV or manual)"""
    print("\n📝 INITIAL SETUP")
    mile_pr_str = input("Enter your mile PR pace (MM:SS): ")
    max_hr_global = float(input("Enter your Max HR (will be used for all runs): "))

    print("\n📊 DATA INPUT OPTIONS")
    print("1. Manual entry (enter each run individually)")
    print("2. Upload CSV file")

    input_method = input("Choose input method (1 or 2): ").strip()

    if input_method == "2":
        # CSV Upload Path
        print(f"\n📁 CSV UPLOAD")
        get_csv_upload_instructions()
        
        print(f"\nExpected CSV format:")
        print(f"   • date: YYYY-MM-DD, MM/DD/YYYY, or similar")
        print(f"   • temp: Temperature in Fahrenheit")
        print(f"   • humidity: Humidity percentage")
        print(f"   • pace: MM:SS format or decimal minutes")
        print(f"   • avg_hr: Average heart rate")
        print(f"   • distance: Distance in miles (optional)")
        print(f"\n💡 Column names will be automatically detected!")
        
        # Try to detect environment and handle accordingly
        csv_df = None
        
        try:
            # Check if in Google Colab
            import google.colab
            print(f"\n📱 Google Colab detected - using file upload widget...")
            csv_df = load_data_from_csv()
            
        except ImportError:
            # Not in Colab, ask for file path
            print(f"\n💻 Local/Jupyter environment detected")
            csv_path = input("Enter CSV file path (or press Enter to browse): ").strip().strip('"')
            
            if not csv_path:
                print("💡 Please provide the file path to your CSV file")
                csv_path = input("File path: ").strip().strip('"')
            
            if csv_path:
                csv_df = load_data_from_csv(csv_path)
        
        if csv_df is not None:
            # Map columns
            column_mapping = map_csv_columns(csv_df)
            
            # Check if we have minimum required columns
            required_minimum = ['date', 'temp', 'humidity', 'pace', 'avg_hr']
            missing_required = [field for field in required_minimum if field not in column_mapping]
            
            if missing_required:
                print(f"\n❌ ERROR: Missing required columns: {missing_required}")
                print(f"Available columns: {list(csv_df.columns)}")
                print(f"Please check your CSV file and column names.")
                print(f"💡 Falling back to manual entry...")
                input_method = "1"
            else:
                # Process the data
                run_data = process_csv_data(csv_df, column_mapping, mile_pr_str, max_hr_global)
                
                if not run_data:
                    print(f"❌ No valid data could be processed from CSV. Please check your file.")
                    print(f"💡 Falling back to manual entry...")
                    input_method = "1"
                else:
                    print(f"\n✅ CSV data loaded successfully!")
                    return run_data, mile_pr_str, max_hr_global
                    
        else:
            print(f"❌ Failed to load CSV. Falling back to manual entry.")
            input_method = "1"

    if input_method == "1":
        # Manual Entry Path
        print("\n📊 MANUAL DATA COLLECTION")
        num_entries = int(input("How many runs do you want to enter? "))
        run_data = collect_manual_data(num_entries, mile_pr_str, max_hr_global)
        return run_data, mile_pr_str, max_hr_global

    # Fallback
    print("\n📊 MANUAL DATA COLLECTION")
    num_entries = int(input("How many runs do you want to enter? "))
    run_data = collect_manual_data(num_entries, mile_pr_str, max_hr_global)
    return run_data, mile_pr_str, max_hr_global
