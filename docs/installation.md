# Installation Guide

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for large datasets)
- Operating System: Windows, macOS, or Linux

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
pip install heat-adaptation-analysis
```

### Method 2: From Source

1. Clone the repository:
```bash
git clone https://github.com/your-username/heat-adaptation-analysis.git
cd heat-adaptation-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv heat_adaptation_env
source heat_adaptation_env/bin/activate  # On Windows: heat_adaptation_env\Scripts\activate
```

3. Install the package:
```bash
pip install -e .
```

### Method 3: Development Installation

For contributors and developers:

```bash
git clone https://github.com/your-username/heat-adaptation-analysis.git
cd heat-adaptation-analysis
pip install -e .[dev]
```

## Dependencies

### Core Dependencies

The following packages are automatically installed:

- **matplotlib** (>=3.5.0) - For data visualization
- **pandas** (>=1.3.0) - For data manipulation
- **numpy** (>=1.21.0) - For numerical computations
- **scikit-learn** (>=1.0.0) - For machine learning models
- **statsmodels** (>=0.13.0) - For statistical analysis
- **scipy** (>=1.7.0) - For scientific computing

### Optional Dependencies

For enhanced functionality:

- **jupyter** - For notebook support
- **ipywidgets** - For interactive widgets in Jupyter
- **plotly** - For interactive visualizations

Install optional dependencies:
```bash
pip install heat-adaptation-analysis[extras]
```

## Platform-Specific Instructions

### Windows

1. Install Python from [python.org](https://python.org)
2. Open Command Prompt or PowerShell as Administrator
3. Follow the pip installation method above

### macOS

1. Install Python using Homebrew:
```bash
brew install python
```
2. Follow the pip installation method above

### Linux (Ubuntu/Debian)

1. Update system packages:
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```
2. Follow the pip installation method above

## Environment Setup

### Google Colab

The package works seamlessly in Google Colab:

```python
!pip install heat-adaptation-analysis
```

### Jupyter Notebook

After installation, start Jupyter:

```bash
jupyter notebook
```

### VS Code

Install the Python extension and select your virtual environment as the interpreter.

## Verification

Test your installation:

```python
import heat_adaptation
print(heat_adaptation.__version__)

# Run a quick test
from heat_adaptation.core.calculations import heat_score
test_score = heat_score(85, 75, 450, 165, 180, 3.1)
print(f"Test heat score: {test_score:.2f}")
```

Expected output:
```
1.0.0
Test heat score: 12.34
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'heat_adaptation'

**Solution:** Ensure you're in the correct virtual environment and the package is installed:
```bash
pip list | grep heat-adaptation
```

#### Permission Denied Errors (Windows)

**Solution:** Run Command Prompt as Administrator or use:
```bash
pip install --user heat-adaptation-analysis
```

#### SSL Certificate Errors

**Solution:** Upgrade pip and certificates:
```bash
pip install --upgrade pip certifi
```

#### Memory Issues with Large Datasets

**Solution:** Increase virtual memory or process data in chunks:
```python
# In your analysis code
import gc
gc.collect()  # Force garbage collection
```

### Getting Help

1. **GitHub Issues**: Report bugs at [GitHub Issues](https://github.com/your-username/heat-adaptation-analysis/issues)
2. **Documentation**: Full API reference at [docs/api_reference.md](api_reference.md)
3. **Examples**: Check the `examples/` directory for usage examples

## Uninstallation

To remove the package:

```bash
pip uninstall heat-adaptation-analysis
```

## Next Steps

After installation, see:
- [Usage Guide](usage.md) - How to use the package
- [API Reference](api_reference.md) - Complete function documentation
- [Examples](../examples/) - Sample code and data