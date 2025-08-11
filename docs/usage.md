# Usage Guide

## Quick Start

The Heat Adaptation Analysis package helps athletes analyze their heat adaptation progress and predict performance improvements. Here's how to get started:

### Basic Usage

```python
from heat_adaptation import HeatAdaptationAnalyzer

# Initialize analyzer
analyzer = HeatAdaptationAnalyzer()

# Load your data
analyzer.load_data('your_running_data.csv')

# Run complete analysis
results = analyzer.analyze()

# Generate visualization
analyzer.visualize()
```

## Data Input Methods

### Method 1: CSV File Upload

The most efficient way to analyze multiple runs:

```python
from heat_adaptation.io import DataManager

# Initialize data manager
data_manager = DataManager()

# Load from CSV
data = data_manager.load_csv('heat_data.csv')

# The CSV should contain these columns:
# date, temp, humidity, pace, avg_hr, distance
```

#### Required CSV Format

| date | temp | humidity | pace | avg_hr | distance |
|------|------|----------|------|--------|----------|
| 2024-07-01 | 85 | 75 | 7:30 | 165 | 3.1 |
| 2024-07-03 | 88 | 80 | 7:45 | 170 | 5.0 |

**Column Requirements:**
- `date`: YYYY-MM-DD, MM/DD/YYYY, or MM-DD-YYYY format
- `temp`: Temperature in Fahrenheit
- `humidity`: Relative humidity percentage (0-100)
- `pace`: MM:SS format (e.g., "7:30") or decimal minutes
- `avg_hr`: Average heart rate during the run
- `distance`: Distance in miles (optional, defaults to 3.1)

### Method 2: Manual Data Entry

For smaller datasets or testing:

```python
from heat_adaptation.data import RunData

# Create individual run entries
runs = [
    RunData(
        date='2024-07-01',
        temp=85,
        humidity=75,
        pace='7:30',
        avg_hr=165,
        distance=3.1
    ),
    RunData(
        date='2024-07-03', 
        temp=88,
        humidity=80,
        pace='7:45',
        avg_hr=170,
        distance=5.0
    )
]

# Add to analyzer
for run in runs:
    analyzer.add_run(run)
```

### Method 3: Dictionary Input

```python
# Define your data as a list of dictionaries
run_data = [
    {
        'date': '2024-07-01',
        'temp': 85,
        'humidity': 75,
        'pace': '7:30',
        'avg_hr': 165,
        'distance': 3.1
    },
    {
        'date': '2024-07-03',
        'temp': 88,
        'humidity': 80,
        'pace': '7:45',
        'avg_hr': 170,
        'distance': 5.0
    }
]

# Load into analyzer
analyzer.load_from_dict(run_data, mile_pr='6:30', max_hr=185)
```

## Complete Analysis Workflow

### Step 1: Initialize and Configure

```python
from heat_adaptation import HeatAdaptationAnalyzer

# Create analyzer with your personal data
analyzer = HeatAdaptationAnalyzer(
    mile_pr='6:30',  # Your mile PR pace
    max_hr=185       # Your maximum heart rate
)
```

### Step 2: Load Your Data

```python
# From CSV file
analyzer.load_csv('my_heat_runs.csv')

# Or load sample data for testing
analyzer.load_sample_data()
```

### Step 3: Run Analysis

```python
# Perform complete analysis
results = analyzer.analyze()

# Access results
print(f"Heat adaptation improvement potential: {results.improvement_percent:.1f}%")
print(f"Days to plateau: {results.plateau_days}")
print(f"Current heat adaptation status: {results.adaptation_category}")
```

### Step 4: Generate Visualizations

```python
# Create the main analysis plot
analyzer.plot_analysis()

# Create detailed subplot analysis
analyzer.plot_detailed()

# Save plots
analyzer.save_plots('my_heat_analysis.png')
```

### Step 5: Get Personalized Advice

```python
# Generate training recommendations
advice = analyzer.get_advice()
print(advice.training_recommendations)
print(advice.timeline_guidance)
```

## Advanced Usage

### Custom Machine Learning Models

```python
from heat_adaptation.models import MLModelTrainer

# Train custom models
trainer = MLModelTrainer()
trainer.train(analyzer.data)

# Use specific model types
trainer.use_model('random_forest')  # or 'gradient_boost'
predictions = trainer.predict(future_conditions)
```

### Feature Engineering

```python
from heat_adaptation.core import FeatureEngineer

# Create enhanced features for ML
engineer = FeatureEngineer()
features = engineer.create_features(analyzer.data)

# Add custom features
features['custom_metric'] = engineer.calculate_heat_index(temp, humidity)
```

### Outlier Detection

```python
# Detect and handle outliers
outliers = analyzer.detect_outliers(method='iqr', threshold=1.5)
analyzer.exclude_outliers(outliers)

# Or use z-score method
outliers = analyzer.detect_outliers(method='zscore', threshold=2.0)
```

### Confidence Intervals

```python
# Get predictions with confidence intervals
predictions = analyzer.predict_with_confidence(confidence_levels=[0.5, 0.8, 0.95])

print(f"Most likely improvement: {predictions.ci_50_lower:.1f}% - {predictions.ci_50_upper:.1f}%")
```

## Platform-Specific Usage

### Google Colab

```python
# Upload CSV file in Colab
from google.colab import files
uploaded = files.upload()

# Load the uploaded file
filename = list(uploaded.keys())[0]
analyzer.load_csv(filename)
```

### Jupyter Notebook

```python
# Interactive widgets for parameter input
from heat_adaptation.widgets import InteractiveAnalyzer

interactive = InteractiveAnalyzer()
interactive.display()  # Shows interactive controls
```

## Example Workflows

### Workflow 1: Season Analysis

Analyze a full season of hot weather training:

```python
# Load season data
analyzer = HeatAdaptationAnalyzer(mile_pr='6:45', max_hr=180)
analyzer.load_csv('summer_2024_runs.csv')

# Analyze progression over time
results = analyzer.analyze_seasonal_progression()

# Generate season report
report = analyzer.generate_season_report()
report.save_pdf('2024_heat_adaptation_report.pdf')
```

### Workflow 2: Race Preparation

Predict race performance in hot conditions:

```python
# Current fitness data
analyzer.load_current_fitness_data()

# Race day conditions
race_conditions = {
    'temp': 92,
    'humidity': 85,
    'distance': 13.1  # Half marathon
}

# Predict race performance
race_prediction = analyzer.predict_race_performance(race_conditions)
print(f"Predicted race HSS: {race_prediction.heat_strain_score:.1f}")
print(f"Recommended pacing adjustment: {race_prediction.pace_adjustment}")
```

### Workflow 3: Training Plan

Generate a heat adaptation training plan:

```python
# Assess current heat fitness
current_fitness = analyzer.assess_heat_fitness()

# Generate 14-day adaptation plan
plan = analyzer.create_adaptation_plan(
    duration_days=14,
    target_improvement=15,
    base_fitness=current_fitness
)

# Export training plan
plan.export_calendar('heat_adaptation_plan.ics')
```

## Understanding Your Results

### Heat Strain Score (HSS)

- **< 10**: Low heat strain (green zone)
- **10-15**: Moderate heat strain (yellow zone)
- **> 15**: High heat strain (red zone)

### Improvement Percentages

- **< 8%**: Already well heat-adapted
- **8-12%**: Moderate adaptation potential
- **12-18%**: Significant adaptation potential
- **> 18%**: High adaptation potential (heat-naive)

### Adaptation Timeline

- **Days 1-3**: Initial physiological responses
- **Days 4-7**: Plasma volume expansion
- **Days 8-12**: Improved sweat rate and cooling
- **Days 10-14**: Peak adaptation achieved

## Troubleshooting Common Issues

### Issue: "Insufficient data for ML model"

**Solution**: You need at least 5 data points. Add more runs or use the research-based model:

```python
analyzer.set_model_preference('research_based')
```

### Issue: "All runs detected as outliers"

**Solution**: Check your data for errors or adjust outlier sensitivity:

```python
analyzer.configure_outlier_detection(method='iqr', threshold=2.0)
```

### Issue: "Unrealistic improvement predictions"

**Solution**: The model applies physiological limits automatically, but you can adjust:

```python
analyzer.set_max_improvement(20.0)  # Cap at 20% improvement
```

## Tips for Best Results

1. **Data Quality**: Ensure accurate temperature and humidity readings
2. **Consistency**: Use the same measurement methods across runs
3. **Minimum Data**: Collect at least 5-10 runs for reliable analysis
4. **Time Span**: Spread runs over several weeks for trend analysis
5. **Conditions**: Include variety in weather conditions
6. **Recovery**: Allow adequate recovery between heat sessions

## Next Steps

- Explore the [API Reference](api_reference.md) for complete function documentation
- Check [Examples](../examples/) for more detailed use cases
- Join our community discussions on GitHub