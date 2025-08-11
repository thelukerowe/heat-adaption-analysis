"""
Heat Adaptation Analysis Package

A comprehensive machine learning-enhanced tool for analyzing and predicting 
heat adaptation in endurance running performance.
"""

from .analysis.data_analyzer import DataAnalyzer
from .io.data_manager import DataManager
from .visualization.visualizer import Visualizer
from .advice.advisor import AdaptationAdvisor
from .core.calculations import HeatCalculations
from .models.ml_models import HeatAdaptationMLModel

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "DataAnalyzer",
    "DataManager", 
    "Visualizer",
    "AdaptationAdvisor",
    "HeatCalculations",
    "HeatAdaptationMLModel",
]


class HeatAdaptationAnalyzer:
    """
    Main interface class for heat adaptation analysis.
    
    This class provides a simple, high-level interface for analyzing
    running performance data and predicting heat adaptation improvements.
    
    Examples:
        >>> analyzer = HeatAdaptationAnalyzer(mile_pr="7:00", max_hr=185)
        >>> analyzer.add_run("2024-07-01", 85, 75, "7:30", 165, 5.0)
        >>> results = analyzer.analyze()
        >>> print(f"Expected improvement: {results.improvement_pct:.1f}%")
    """
    
    def __init__(self, mile_pr: str, max_hr: int):
        """
        Initialize the analyzer.
        
        Args:
            mile_pr: Mile PR pace in MM:SS format (e.g., "6:30")
            max_hr: Maximum heart rate in BPM
        """
        self.data_manager = DataManager(mile_pr, max_hr)
        self.analyzer = DataAnalyzer()
        self.visualizer = Visualizer()
        self.advisor = AdaptationAdvisor()
        self.run_data = []
    
    def add_run(self, date: str, temp: float, humidity: float, 
                pace: str, avg_hr: float, distance: float):
        """
        Add a single run to the analysis.
        
        Args:
            date: Date in YYYY-MM-DD format
            temp: Temperature in Fahrenheit
            humidity: Humidity percentage
            pace: Pace in MM:SS format per mile
            avg_hr: Average heart rate
            distance: Distance in miles
        """
        run = self.data_manager.create_run_entry(
            date, temp, humidity, pace, avg_hr, distance
        )
        self.run_data.append(run)
    
    def load_from_csv(self, csv_path: str):
        """Load run data from CSV file."""
        self.run_data = self.data_manager.load_from_csv(csv_path)
    
    @classmethod
    def from_csv(cls, csv_path: str, mile_pr: str = None, max_hr: int = None):
        """
        Create analyzer instance from CSV file.
        
        Args:
            csv_path: Path to CSV file
            mile_pr: Mile PR pace (will prompt if not provided)
            max_hr: Max heart rate (will prompt if not provided)
        """
        if mile_pr is None:
            mile_pr = input("Enter your mile PR pace (MM:SS): ").strip()
        if max_hr is None:
            max_hr = int(input("Enter your Max HR (BPM): ").strip())
        
        analyzer = cls(mile_pr, max_hr)
        analyzer.load_from_csv(csv_path)
        return analyzer
    
    def analyze(self):
        """
        Analyze the run data and return results.
        
        Returns:
            AnalysisResults: Complete analysis results including predictions
        """
        if not self.run_data:
            raise ValueError("No run data available for analysis")
        
        return self.analyzer.analyze(self.run_data)
    
    def visualize(self, results=None, save_path=None):
        """
        Create visualization of results.
        
        Args:
            results: Analysis results (will analyze if not provided)
            save_path: Path to save plot (optional)
        """
        if results is None:
            results = self.analyze()
        
        return self.visualizer.create_comprehensive_plot(results, save_path)
    
    def get_advice(self, results=None):
        """
        Get personalized heat adaptation advice.
        
        Args:
            results: Analysis results (will analyze if not provided)
        """
        if results is None:
            results = self.analyze()
        
        return self.advisor.generate_advice(results)