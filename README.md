# Heat Adaptation Analysis

A comprehensive machine-learning enhanced tool for analyzing and predicting heat adaptation in endurance running performance. [click here](https://heat-adaption-analysis-heyiszcrfsappbnpappwlz.streamlit.app/)

## Overview

This tool analyzes running performance in hot conditions and uses machine learning models to predict how much your performance will improve after proper heat adaptation training. It provides personalized training recommendations based on your current heat tolerance and adaptation potential.

## Features

- **ML-Enhanced Analysis**: Uses Random Forest and Gradient Boosting models for accurate predictions
- **Heat Strain Score (HSS)**: Comprehensive metric combining temperature, humidity, pace, heart rate, and distance
- **Adaptation Prediction**: Predicts performance improvement after heat adaptation training
- **Confidence Intervals**: Statistical confidence ranges for predictions
- **Outlier Detection**: Automatically identifies and handles outlier performances
- **Multiple Data Input**: Manual entry or CSV file upload
- **Personalized Advice**: Custom training recommendations based on your adaptation potential

## Installation

```bash
git clone https://github.com/yourusername/heat-adaptation-analysis.git
cd heat-adaptation-analysis
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install heat-adaptation-analysis
```

## Quick Start

### Basic Usage
```python
from heat_adaptation import HeatAdaptationAnalyzer

# Initialize analyzer
analyzer = HeatAdaptationAnalyzer(mile_pr="7:00", max_hr=185)

# Add run data
analyzer.add_run(
    date="2024-07-01",
    temp=85,
    humidity=75,
    pace="7:30",
    avg_hr=165,
    distance=5.0
)

# Analyze and predict
results = analyzer.analyze()
print(f"Expected improvement: {results.improvement_pct:.1f}%")
```

### CSV Data Import
```python
# Load from CSV file
analyzer = HeatAdaptationAnalyzer.from_csv("my_running_data.csv")
results = analyzer.analyze()
```

## Data Format

### Manual Entry
Required for each run:
- Date (YYYY-MM-DD format)
- Temperature (°F)
- Humidity (%)
- Pace (MM:SS per mile)
- Average Heart Rate (bpm)
- Distance (miles)

### CSV Format
Expected columns (auto-detected):
- `date`: Date of run
- `temp`: Temperature in Fahrenheit
- `humidity`: Humidity percentage
- `pace`: Pace in MM:SS format
- `avg_hr`: Average heart rate
- `distance`: Distance in miles

See `examples/sample_data.csv` for reference.

## Model Types

The system automatically selects the best model based on data availability:

1. **Machine Learning** (10+ clean data points): Random Forest + Gradient Boosting
2. **Complex Logarithmic** (5-9 data points): Advanced statistical modeling
3. **Research-Based** (3-4 data points): Physiological research-based predictions

## Heat Strain Score (HSS)

HSS combines multiple factors into a single metric:
- Temperature and humidity (with interaction effects)
- Running pace relative to one-mile personal best
- Heart rate as percentage of maximum
- Distance scaling factor

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this tool in research, please cite:
```
Heat Adaptation Analysis Tool (2024)
https://github.com/yourusername/heat-adaptation-analysis
```

## Acknowledgments

- Physiological models based on heat adaptation research
- Statistical methods from sports science literature
- Machine learning approaches for performance prediction
