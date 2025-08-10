#!/usr/bin/env python3
"""
Main entry point for the Heat Adaptation Analysis tool.

This script provides a command-line interface for analyzing running performance
in hot conditions and predicting heat adaptation improvements.
"""

import sys
import argparse
from datetime import datetime
from heat_adaptation.analysis.data_analyzer import DataAnalyzer
from heat_adaptation.io.data_manager import DataManager
from heat_adaptation.visualization.visualizer import Visualizer
from heat_adaptation.advice.advisor import AdaptationAdvisor


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="Heat Adaptation Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --interactive          # Interactive mode
  python main.py --csv data.csv --mile-pr 7:00 --max-hr 185
  python main.py --help                 # Show this help
        """
    )
    
    parser.add_argument(
        "--csv", 
        type=str, 
        help="Path to CSV file with running data"
    )
    parser.add_argument(
        "--mile-pr", 
        type=str, 
        help="Mile PR pace in MM:SS format (e.g., 6:30)"
    )
    parser.add_argument(
        "--max-hr", 
        type=int, 
        help="Maximum heart rate (BPM)"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true", 
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Output file path for visualization (optional)"
    )
    parser.add_argument(
        "--no-plot", 
        action="store_true", 
        help="Skip visualization"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version="Heat Adaptation Analysis v1.0.0"
    )
    
    args = parser.parse_args()
    
    try:
        if args.interactive or (not args.csv and not args.mile_pr and not args.max_hr):
            # Run interactive mode
            run_interactive_mode()
        else:
            # Run with command line arguments
            if not args.mile_pr or not args.max_hr:
                print("Error: --mile-pr and --max-hr are required when not in interactive mode")
                sys.exit(1)
            
            run_batch_mode(args)
            
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


def run_interactive_mode():
    """Run the tool in interactive mode."""
    print("🔥 Heat Adaptation Analysis v1.0.0")
    print("=" * 55)
    
    # Get basic parameters
    mile_pr_str = input("\nEnter your mile PR pace (MM:SS): ").strip()
    max_hr = int(input("Enter your Max HR (BPM): ").strip())
    
    # Initialize components
    data_manager = DataManager(mile_pr_str, max_hr)
    
    # Choose data input method
    print("\n📊 DATA INPUT OPTIONS")
    print("1. Manual entry (enter each run individually)")
    print("2. Upload CSV file")
    
    choice = input("Choose input method (1 or 2): ").strip()
    
    if choice == "2":
        csv_path = input("Enter CSV file path: ").strip().strip('"')
        try:
            run_data = data_manager.load_from_csv(csv_path)
        except Exception as e:
            print(f"Error loading CSV: {e}")
            print("Falling back to manual entry...")
            run_data = data_manager.collect_manual_data()
    else:
        run_data = data_manager.collect_manual_data()
    
    if not run_data:
        print("No data collected. Exiting.")
        return
    
    # Analyze data
    analyzer = DataAnalyzer()
    results = analyzer.analyze(run_data)
    
    # Generate visualization
    print("\n📊 Generating visualization...")
    visualizer = Visualizer()
    try:
        visualizer.create_comprehensive_plot(results)
        print("✅ Visualization complete!")
    except Exception as e:
        print(f"Warning: Visualization failed: {e}")
    
    # Generate advice
    advisor = AdaptationAdvisor()
    advisor.generate_advice(results)


def run_batch_mode(args):
    """Run the tool in batch mode with command line arguments."""
    print("🔥 Heat Adaptation Analysis - Batch Mode")
    
    # Initialize data manager
    data_manager = DataManager(args.mile_pr, args.max_hr)
    
    # Load data
    if args.csv:
        try:
            run_data = data_manager.load_from_csv(args.csv)
            print(f"✅ Loaded {len(run_data)} runs from {args.csv}")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            sys.exit(1)
    else:
        print("Error: CSV file required for batch mode")
        sys.exit(1)
    
    # Analyze data
    analyzer = DataAnalyzer()
    results = analyzer.analyze(run_data)
    
    # Print summary
    print(f"\n📊 ANALYSIS SUMMARY")
    print(f"=" * 40)
    print(f"Total runs analyzed: {len(run_data)}")
    print(f"Model used: {results.model_used.replace('_', ' ').title()}")
    if results.improvement_pct is not None:
        print(f"Expected improvement: {results.improvement_pct:.1f}%")
        print(f"Estimated plateau days: {results.plateau_days}")
    else:
        print("Insufficient data for prediction")
    
    # Generate visualization if requested
    if not args.no_plot:
        print("\n📊 Generating visualization...")
        visualizer = Visualizer()
        try:
            if args.output:
                visualizer.create_comprehensive_plot(results, save_path=args.output)
                print(f"✅ Visualization saved to {args.output}")
            else:
                visualizer.create_comprehensive_plot(results)
                print("✅ Visualization displayed")
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    # Generate advice
    advisor = AdaptationAdvisor()
    advisor.generate_advice(results)


if __name__ == "__main__":
    main()