"""
Example Usage of Heat Adaptation Analysis

Demonstrates how to use the modularized heat adaptation analysis package
with both CSV and manual data input methods.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the parent directory to the path to import heat_adaptation package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from heat_adaptation.io import get_data_input
from heat_adaptation.core import create_ml_features
from heat_adaptation.analysis import detect_outliers, estimate_threshold, analyze_data_quality
from heat_adaptation.models import create_prediction_engine
from heat_adaptation.visualization import create_original_visualization
from heat_adaptation.advice import generate_adaptation_advice


def run_csv_example():
    """Example using the provided sample CSV data"""
    print("🔥 Heat Adaptation Analysis - CSV Example")
    print("=" * 50)
    
    # Load sample data
    csv_path = os.path.join(os.path.dirname(__file__), 'sample_data.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Sample CSV not found at: {csv_path}")
        print("Please ensure sample_data.csv is in the examples/ directory")
        return
    
    print(f"📁 Loading sample data from: {csv_path}")
    
    # Simulate CSV input by setting up the data manually
    from heat_adaptation.io.data_manager import load_data_from_csv, map_csv_columns, process_csv_data
    
    # Load and process CSV
    df = load_data_from_csv(csv_path)
    if df is None:
        print("❌ Failed to load CSV data")
        return
    
    # Set up user parameters (normally from input)
    mile_pr_str = "6:30"  # Example PR pace
    max_hr_global = 190.0  # Example max HR
    
    # Map columns and process data
    column_mapping = map_csv_columns(df)
    run_data = process_csv_data(df, column_mapping, mile_pr_str, max_hr_global)
    
    if not run_data:
        print("❌ No valid data processed")
        return
    
    # Run complete analysis
    run_complete_analysis(run_data, mile_pr_str, max_hr_global)


def run_manual_example():
    """Example showing manual data entry (simulated)"""
    print("🔥 Heat Adaptation Analysis - Manual Entry Example")
    print("=" * 55)
    
    # Simulate manual entry with example data
    from datetime import datetime
    from heat_adaptation.core.calculations import heat_score
    from heat_adaptation.io.data_manager import pace_to_seconds
    
    mile_pr_str = "6:45"
    max_hr_global = 185.0
    mile_pr_sec = pace_to_seconds(mile_pr_str)
    
    # Example manual data
    manual_runs = [
        {"date": "2024-07-15", "temp": 88, "humidity": 78, "pace": "7:20", "avg_hr": 168, "distance": 5.0},
        {"date": "2024-07-17", "temp": 91, "humidity": 82, "pace": "7:35", "avg_hr": 172, "distance": 3.1},
        {"date": "2024-07-20", "temp": 94, "humidity": 85, "pace": "7:50", "avg_hr": 176, "distance": 4.0},
        {"date": "2024-07-22", "temp": 87, "humidity": 75, "pace": "7:15", "avg_hr": 165, "distance": 6.0},
        {"date": "2024-07-25", "temp": 95, "humidity": 88, "pace": "8:05", "avg_hr": 178, "distance": 3.1}
    ]
    
    run_data = []
    for run in manual_runs:
        date_obj = datetime.strptime(run["date"], "%Y-%m-%d")
        pace_sec = pace_to_seconds(run["pace"])
        raw_score = heat_score(run["temp"], run["humidity"], pace_sec, 
                              run["avg_hr"], max_hr_global, run["distance"], multiplier=1.0)
        
        run_data.append({
            'date': date_obj,
            'temp': run["temp"],
            'humidity': run["humidity"],
            'pace_sec': pace_sec,
            'avg_hr': run["avg_hr"],
            'max_hr': max_hr_global,
            'distance': run["distance"],
            'raw_score': raw_score,
            'adjusted_score': None,
            'relative_score': None,
            'mile_pr_sec': mile_pr_sec
        })
    
    print(f"📊 Created {len(run_data)} example runs")
    
    # Run complete analysis
    run_complete_analysis(run_data, mile_pr_str, max_hr_global)


def run_complete_analysis(run_data, mile_pr_str, max_hr_global):
    """Run the complete heat adaptation analysis pipeline"""
    from heat_adaptation.core.calculations import adjust_multiplier, heat_score
    from heat_adaptation.io.data_manager import pace_to_seconds
    
    # Extract basic data
    raw_scores = [run['raw_score'] for run in run_data]
    mile_pr_sec = pace_to_seconds(mile_pr_str)
    
    # Analyze data quality
    print(f"\n📊 DATA QUALITY ANALYSIS")
    quality_stats = analyze_data_quality(run_data)
    print(f"   Total runs: {quality_stats['total_runs']}")
    print(f"   Outliers: {quality_stats['outlier_count']} ({quality_stats['outlier_percentage']:.1f}%)")
    print(f"   Mean HSS: {quality_stats['mean_hss']:.2f}")
    print(f"   Clean mean HSS: {quality_stats['clean_mean_hss']:.2f}")
    
    # Convert to DataFrame for ML processing
    print(f"\n🔄 CONVERTING TO ML FORMAT...")
    df = pd.DataFrame(run_data)
    df_features = create_ml_features(df)
    print(f"   ✅ Created {len(df_features.columns)} features for ML analysis")
    
    # Detect outliers and calculate threshold
    print(f"\n🔍 OUTLIER DETECTION & THRESHOLD CALCULATION")
    outliers = detect_outliers(raw_scores, method='iqr', threshold=1.5)
    clean_scores = [score for i, score in enumerate(raw_scores) if not outliers[i]]
    
    if len(clean_scores) > 0:
        threshold = estimate_threshold(clean_scores)
        print(f"   Outlier Detection: {sum(outliers)} outlier(s) detected and excluded from threshold calculation")
        if sum(outliers) > 0:
            outlier_dates = [run_data[i]['date'].strftime('%Y-%m-%d') for i in range(len(outliers)) if outliers[i]]
            print(f"   Outlier runs on: {', '.join(outlier_dates)}")
    else:
        threshold = estimate_threshold(raw_scores)
        print("   ⚠️  Warning: All runs detected as outliers. Using all data for threshold.")
    
    print(f"   📊 Calculated threshold: {threshold:.2f}")
    
    # Calculate adjusted and relative scores
    print(f"\n⚙️  CALCULATING ADJUSTED SCORES...")
    for run in run_data:
        multiplier = adjust_multiplier(run['raw_score'], threshold, run['pace_sec'], mile_pr_sec)
        adjusted_hss = heat_score(run['temp'], run['humidity'], run['pace_sec'], 
                                 run['avg_hr'], run['max_hr'], run['distance'], multiplier)
        relative_score = adjusted_hss / threshold if threshold != 0 else 0
        
        run['adjusted_score'] = adjusted_hss
        run['relative_score'] = relative_score
    
    # Update DataFrame with adjusted scores
    df['adjusted_score'] = [run['adjusted_score'] for run in run_data]
    df['relative_score'] = [run['relative_score'] for run in run_data]
    df_features['adjusted_score'] = df['adjusted_score']
    
    # Prepare arrays for prediction
    dates = np.array([run['date'] for run in run_data])
    raw_scores_arr = np.array([run['raw_score'] for run in run_data])
    adjusted_scores = np.array([run['adjusted_score'] for run in run_data])
    
    # Sort data by date
    sort_idx = np.argsort(dates)
    dates = dates[sort_idx]
    raw_scores_arr = raw_scores_arr[sort_idx]
    adjusted_scores = adjusted_scores[sort_idx]
    
    # Generate predictions using modularized prediction engine
    prediction_engine = create_prediction_engine()
    
    model_result = prediction_engine.generate_predictions(
        run_data=run_data,
        df_features=df_features,
        outliers=outliers,
        threshold=threshold,
        dates=dates,
        adjusted_scores=adjusted_scores
    )
    
    if model_result:
        # Print prediction summary
        prediction_engine.print_prediction_summary(run_data, outliers)
        
        # Generate personalized advice
        generate_adaptation_advice(
            model_result.improvement_pct,
            model_result.baseline_hss,
            model_result.plateau_days,
            run_data
        )
        
        # Create visualization
        ci_50_lower, ci_50_upper = model_result.confidence_intervals['50']
        ci_80_lower, ci_80_upper = model_result.confidence_intervals['80']
        ci_95_lower, ci_95_upper = model_result.confidence_intervals['95']
        
        print(f"\n📊 CREATING VISUALIZATION...")
        create_original_visualization(
            run_data, dates, raw_scores_arr, adjusted_scores,
            model_result.adapted_runs, model_result.adapted_dates, outliers,
            model_result.improvement_pct, threshold, model_result.model_type,
            ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper,
            ci_95_lower, ci_95_upper, model_result.plateau_days
        )
        
        print(f"   ✅ Analysis complete!")
        
    else:
        print(f"\n⚠️ No predictions available due to insufficient data.")


def main():
    """Main function to run examples"""
    print("🤖 Heat Adaptation Analysis - Example Usage")
    print("=" * 50)
    print("Choose an example to run:")
    print("1. CSV Example (using sample_data.csv)")
    print("2. Manual Entry Example (simulated)")
    print("3. Interactive Mode (full user input)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        run_csv_example()
    elif choice == "2":
        run_manual_example()
    elif choice == "3":
        # Use the full interactive data input
        run_data, mile_pr_str, max_hr_global = get_data_input()
        run_complete_analysis(run_data, mile_pr_str, max_hr_global)
    else:
        print("Invalid choice. Running CSV example by default...")
        run_csv_example()


if __name__ == "__main__":
    main()
