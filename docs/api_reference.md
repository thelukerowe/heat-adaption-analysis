# API Reference

## Core Classes

### HeatAdaptationAnalyzer

The main analysis class that orchestrates the entire heat adaptation analysis.

```python
class HeatAdaptationAnalyzer:
    def __init__(self, mile_pr: str = None, max_hr: float = None)
```

#### Parameters
- `mile_pr` (str): Mile personal record in MM:SS format (e.g., "6:30")
- `max_hr` (float): Maximum heart rate in beats per minute

#### Methods

##### `load_csv(filepath: str) -> bool`
Load running data from a CSV file.

**Parameters:**
- `filepath` (str): Path to the CSV file

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
success = analyzer.load_csv('my_runs.csv')
if success:
    print("Data loaded successfully")
```

##### `analyze() -> AnalysisResults`
Perform complete heat adaptation analysis.

**Returns:**
- `AnalysisResults`: Object containing all analysis results

**Example:**
```python
results = analyzer.analyze()
print(f"Improvement potential: {results.improvement_percent}%")
```

##### `plot_analysis() -> None`
Generate the main analysis visualization.

**Example:**
```python
analyzer.plot_analysis()
plt.show()
```

##### `get_advice() -> AdaptationAdvice`
Get personalized heat adaptation training advice.

**Returns:**
- `AdaptationAdvice`: Object containing training recommendations

---

## Core Module (`heat_adaptation.core`)

### Calculations

#### `heat_score(temp: float, humidity: float, pace_sec_per_mile: float, avg_hr: float, max_hr: float, distance: float, multiplier: float = 1.0) -> float`

Calculate the Heat Strain Score (HSS) for a running session.

**Parameters:**
- `temp` (float): Temperature in Fahrenheit
- `humidity` (float): Relative humidity percentage (0-100)
- `pace_sec_per_mile` (float): Pace in seconds per mile
- `avg_hr` (float): Average heart rate
- `max_hr` (float): Maximum heart rate
- `distance` (float): Distance in miles
- `multiplier` (float, optional): Adjustment multiplier (default: 1.0)

**Returns:**
- `float`: Heat Strain Score

**Formula:**
The HSS uses a complex formula incorporating:
- Temperature normalization (20-110°F range)
- Humidity normalization with exponential scaling
- Heart rate efficiency (avg_hr/max_hr)
- Pace effort relative to threshold
- Distance scaling factor

**Example:**
```python
from heat_adaptation.core.calculations import heat_score

hss = heat_score(
    temp=85,
    humidity=75,
    pace_sec_per_mile=450,  # 7:30 pace
    avg_hr=165,
    max_hr=185,
    distance=3.1
)
print(f"Heat Strain Score: {hss:.2f}")
```

#### `calculate_heat_index(temp_f: float, humidity_pct: float) -> float`

Calculate the heat index (feels-like temperature).

**Parameters:**
- `temp_f` (float): Temperature in Fahrenheit
- `humidity_pct` (float): Humidity percentage

**Returns:**
- `float`: Heat index in Fahrenheit

**Example:**
```python
heat_index = calculate_heat_index(85, 75)
print(f"Feels like: {heat_index:.1f}°F")
```

#### `pace_to_seconds(pace_str: str) -> int`

Convert pace string to seconds per mile.

**Parameters:**
- `pace_str` (str): Pace in MM:SS format

**Returns:**
- `int`: Pace in seconds per mile

**Example:**
```python
seconds = pace_to_seconds("7:30")
# Returns: 450
```

#### `apply_physiological_limits(improvement_pct: float, max_improvement: float = 25.0) -> float`

Apply realistic physiological limits to predicted improvements.

**Parameters:**
- `improvement_pct` (float): Raw improvement percentage
- `max_improvement` (float, optional): Maximum allowed improvement (default: 25.0)

**Returns:**
- `float`: Capped improvement percentage

### Feature Engineering

#### `create_ml_features(df: pd.DataFrame) -> pd.DataFrame`

Create enhanced features for machine learning models.

**Parameters:**
- `df` (pd.DataFrame): Raw running data

**Returns:**
- `pd.DataFrame`: Enhanced feature matrix

**Features Created:**
- `heat_index`: Calculated heat index
- `pace_vs_pr`: Pace relative to personal record
- `hr_efficiency`: Heart rate as percentage of max
- `temp_humidity_interaction`: Temperature × humidity interaction
- `environmental_stress`: Combined environmental stress metric
- `performance_stress`: Combined performance stress metric
- Rolling averages and trend indicators

---

## Models Module (`heat_adaptation.models`)

### HeatAdaptationMLModel

Machine learning model trainer and predictor for heat adaptation analysis.

```python
class HeatAdaptationMLModel:
    def __init__(self)
```

#### Methods

##### `train_models(features_df: pd.DataFrame, target_col: str = 'raw_score', min_samples: int = 5) -> dict`

Train multiple ML models and select the best performer.

**Parameters:**
- `features_df` (pd.DataFrame): Feature matrix with target column
- `target_col` (str, optional): Target column name (default: 'raw_score')
- `min_samples` (int, optional): Minimum samples required for training (default: 5)

**Returns:**
- `dict`: Model performance metrics or None if insufficient data

**Available Models:**
- Random Forest Regressor
- Gradient Boosting Regressor

**Example:**
```python
ml_model = HeatAdaptationMLModel()
performance = ml_model.train_models(feature_data)
if performance:
    print(f"Best model RMSE: {performance['rmse']:.2f}")
```

##### `predict(features_df: pd.DataFrame) -> np.ndarray`

Make predictions using the trained model.

**Parameters:**
- `features_df` (pd.DataFrame): Features for prediction

**Returns:**
- `np.ndarray`: Predictions or None if model not trained

##### `print_feature_importance() -> None`

Print the top 5 most important features for interpretation.

---

## Data Module (`heat_adaptation.data`)

### RunData

Data class representing a single running session.

```python
@dataclass
class RunData:
    date: str
    temp: float
    humidity: float
    pace: str
    avg_hr: float
    distance: float = 3.1
```

#### Methods

##### `to_dict() -> dict`

Convert to dictionary format.

##### `from_dict(data: dict) -> 'RunData'`

Create RunData instance from dictionary.

**Example:**
```python
run = RunData(
    date='2024-07-01',
    temp=85,
    humidity=75,
    pace='7:30',
    avg_hr=165,
    distance=3.1
)
```

---

## IO Module (`heat_adaptation.io`)

### DataManager

Handles data input/output operations.

```python
class DataManager:
    def __init__(self)
```

#### Methods

##### `load_csv(filepath: str) -> pd.DataFrame`

Load data from CSV file with intelligent column mapping.

**Parameters:**
- `filepath` (str): Path to CSV file

**Returns:**
- `pd.DataFrame`: Loaded and processed data

##### `map_csv_columns(df: pd.DataFrame) -> dict`

Automatically map CSV columns to required fields.

**Parameters:**
- `df` (pd.DataFrame): Raw CSV data

**Returns:**
- `dict`: Column mapping dictionary

**Supported Column Names:**
- **Date**: 'date', 'Date', 'DATE', 'run_date', 'workout_date'
- **Temperature**: 'temp', 'temperature', 'Temperature', 'TEMP', 'temp_f'
- **Humidity**: 'humidity', 'Humidity', 'HUMIDITY', 'humid', 'rh', 'RH'
- **Pace**: 'pace', 'Pace', 'PACE', 'avg_pace', 'average_pace'
- **Heart Rate**: 'avg_hr', 'average_hr', 'hr', 'heart_rate', 'HR'
- **Distance**: 'distance', 'Distance', 'miles', 'km', 'dist'

##### `create_sample_csv(filename: str = 'sample_heat_data.csv') -> None`

Create a sample CSV template.

**Parameters:**
- `filename` (str, optional): Output filename (default: 'sample_heat_data.csv')

---

## Analysis Module (`heat_adaptation.analysis`)

### DataAnalyzer

Performs statistical analysis on running data.

#### Methods

##### `detect_outliers(scores: list, method: str = 'iqr', threshold: float = 1.5) -> list`

Detect outliers in heat strain scores.

**Parameters:**
- `scores` (list): Heat strain scores
- `method` (str, optional): Detection method - 'iqr' or 'zscore' (default: 'iqr')
- `threshold` (float, optional): Threshold for outlier detection (default: 1.5)

**Returns:**
- `list`: Boolean list indicating outliers

##### `calculate_confidence_intervals(predictions: np.ndarray, residuals: np.ndarray, confidence_level: float = 0.95) -> tuple`

Calculate confidence intervals for predictions.

**Parameters:**
- `predictions` (np.ndarray): Model predictions
- `residuals` (np.ndarray): Model residuals
- `confidence_level` (float, optional): Confidence level (default: 0.95)

**Returns:**
- `tuple`: (ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper, ci_95_lower, ci_95_upper)

##### `check_plateau(smoothed_scores: list, threshold_pct: float = 0.01, days: int = 3) -> bool`

Check if adaptation has reached a plateau.

**Parameters:**
- `smoothed_scores` (list): Smoothed heat strain scores
- `threshold_pct` (float, optional): Improvement threshold (default: 0.01)
- `days` (int, optional): Consecutive days required (default: 3)

**Returns:**
- `bool`: True if plateau detected

---

## Visualization Module (`heat_adaptation.visualization`)

### Visualizer

Creates analysis visualizations.

```python
class Visualizer:
    def __init__(self, analyzer: HeatAdaptationAnalyzer)
```

#### Methods

##### `plot_comprehensive_analysis() -> None`

Create the main comprehensive analysis plot showing current vs adapted performance.

**Features:**
- Current heat strain scores with outlier detection
- Adjusted and relative scores
- Future adapted performance predictions
- Confidence intervals (50%, 80%, 95%)
- Color-coded risk levels

##### `plot_detailed_subplots() -> None`

Create detailed 4-subplot analysis:
1. Heat strain scores over time
2. Environmental conditions
3. Performance metrics
4. Adaptation predictions

##### `risk_color_hss(score: float) -> str`

Get risk color for heat strain score.

**Parameters:**
- `score` (float): Heat strain score

**Returns:**
- `str`: Color name ('green', 'orange', 'red')

**Color Mapping:**
- Green: HSS < 10 (low risk)
- Orange: HSS 10-15 (moderate risk)  
- Red: HSS > 15 (high risk)

---

## Advice Module (`heat_adaptation.advice`)

### AdaptationAdvisor

Generates personalized heat adaptation training advice.

```python
class AdaptationAdvisor:
    def __init__(self)
```

#### Methods

##### `generate_advice(improvement_pct: float, baseline_hss: float, plateau_days: int, run_data: list) -> AdaptationAdvice`

Generate comprehensive training recommendations.

**Parameters:**
- `improvement_pct` (float): Expected improvement percentage
- `baseline_hss` (float): Current baseline heat strain score
- `plateau_days` (int): Days to reach adaptation plateau
- `run_data` (list): Historical running data

**Returns:**
- `AdaptationAdvice`: Structured advice object

##### `categorize_adaptation_status(improvement_pct: float) -> str`

Categorize current heat adaptation status.

**Parameters:**
- `improvement_pct` (float): Improvement potential percentage

**Returns:**
- `str`: Adaptation category

**Categories:**
- "Well Heat-Adapted" (< 8% improvement)
- "Somewhat Heat-Adapted" (8-12% improvement) 
- "Moderately Heat-Adapted" (12-18% improvement)
- "Heat-Naive" (≥ 18% improvement)

---

## Result Classes

### AnalysisResults

Contains complete analysis results.

```python
@dataclass
class AnalysisResults:
    improvement_percent: float
    plateau_days: int
    adaptation_category: str
    current_hss_mean: float
    adapted_hss_mean: float
    outliers_detected: int
    model_used: str
    confidence_intervals: dict
    feature_importance: dict
```

### AdaptationAdvice

Contains personalized training advice.

```python
@dataclass  
class AdaptationAdvice:
    category: str
    training_recommendations: list
    timeline_guidance: dict
    precautions: list
    monitoring_advice: list
```

---

## Constants and Enums

### Model Types

```python
class ModelType(Enum):
    MACHINE_LEARNING = "machine_learning"
    COMPLEX_LOGARITHMIC = "complex_logarithmic" 
    RESEARCH_BASED = "research_based"
    INSUFFICIENT_DATA = "insufficient_data"
```

### Risk Levels

```python
class RiskLevel(Enum):
    LOW = "low"      # HSS < 10
    MODERATE = "moderate"  # HSS 10-15
    HIGH = "high"    # HSS > 15
```

---

## Utility Functions

### Normalization

#### `normalize(value: float, min_val: float, max_val: float) -> float`

Normalize value to 0-1 range.

#### `scale_distance(distance: float, min_dist: float = 1, max_dist: float = 15) -> float`

Scale distance for heat score calculation.

### Statistical

#### `loess_smooth(x: np.ndarray, y: np.ndarray, frac: float = 0.3) -> np.ndarray`

Apply LOESS smoothing to data.

#### `estimate_threshold(scores: list) -> float`

Estimate threshold using median of scores.

---

## Error Handling

### Custom Exceptions

#### `InsufficientDataError`
Raised when insufficient data is provided for analysis.

#### `InvalidDataFormatError` 
Raised when data format is invalid.

#### `ModelTrainingError`
Raised when ML model training fails.

**Example Error Handling:**
```python
try:
    results = analyzer.analyze()
except InsufficientDataError:
    print("Please provide at least 5 data points for analysis")
except InvalidDataFormatError as e:
    print(f"Data format error: {e}")
```

---

## Configuration

### Default Settings

```python
DEFAULT_CONFIG = {
    'min_data_points_ml': 10,
    'min_data_points_complex': 30, 
    'min_data_points_simple': 5,
    'max_improvement_percent': 25.0,
    'outlier_threshold': 1.5,
    'confidence_levels': [0.5, 0.8, 0.95]
}
```

### Customization

```python
# Customize analysis parameters
analyzer.configure(
    outlier_threshold=2.0,
    max_improvement=20.0,
    confidence_levels=[0.6, 0.9]
)
```