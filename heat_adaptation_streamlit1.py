import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import io

# ML IMPORTS
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Streamlit page configuration
st.set_page_config(
    page_title="Heat Adaptation Analysis",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #4CAF50;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# --- All your original functions (keeping them exactly the same) ---

def apply_physiological_limits(improvement_pct, max_improvement=25.0):
    """Apply realistic physiological limits to heat adaptation improvement"""
    if improvement_pct is None:
        return None
    
    # Cap improvement at physiologically realistic levels
    limited_improvement = min(abs(improvement_pct), max_improvement)
    
    # Add some context based on improvement magnitude
    if limited_improvement > 20:
        st.info(f"Note: Raw model predicted {abs(improvement_pct):.1f}% improvement, capped at {limited_improvement:.1f}% (physiological maximum)")
    elif limited_improvement > 15:
        st.info(f"Note: {limited_improvement:.1f}% improvement suggests you're relatively heat-naive")
    elif limited_improvement > 10:
        st.info(f"Note: {limited_improvement:.1f}% improvement suggests moderate heat adaptation potential")
    else:
        st.info(f"Note: {limited_improvement:.1f}% improvement suggests you're already well heat-adapted")
    
    return limited_improvement

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)

def scale_distance(distance, min_dist=1, max_dist=15):
    norm_dist = normalize(distance, min_dist, max_dist)
    return 0.5 + (norm_dist * 0.5)

def pace_to_seconds(pace_str):
    if isinstance(pace_str, str) and ':' in pace_str:
        minutes, seconds = map(int, pace_str.split(":"))
        return minutes * 60 + seconds
    else:
        # Handle decimal minutes format
        pace_min = float(pace_str)
        return int(pace_min * 60)

def heat_score(temp, humidity, pace_sec_per_mile, avg_hr, max_hr, distance, multiplier=1.0):
    T_norm = normalize(temp, 20, 110)
    H_norm = normalize(humidity, 20, 99)
    dist_scaled = scale_distance(distance)

    score = multiplier * (
        ((2 * T_norm) + (1.2 * (H_norm ** 1.3)) + (4.2 * (T_norm * H_norm))) *
        ((avg_hr / max_hr) ** 1.1) *
        ((600 / pace_sec_per_mile) ** 1.3) *
        (dist_scaled ** 0.4)
    )
    return score

def create_ml_features(df):
    """Create enhanced features for machine learning models"""
    features_df = df.copy()
    
    # Basic engineered features
    features_df['heat_index'] = calculate_heat_index(df['temp'], df['humidity'])
    features_df['pace_vs_pr'] = df['pace_sec'] / df['mile_pr_sec']
    features_df['hr_efficiency'] = df['avg_hr'] / df['max_hr']
    features_df['temp_humidity_interaction'] = df['temp'] * df['humidity'] / 100
    features_df['pace_hr_interaction'] = features_df['pace_vs_pr'] * features_df['hr_efficiency']
    
    # Time-based features
    if len(df) >= 3:
        features_df['hss_rolling_3'] = df['raw_score'].rolling(window=min(3, len(df)), min_periods=1).mean()
        features_df['temp_rolling_3'] = df['temp'].rolling(window=min(3, len(df)), min_periods=1).mean()
        features_df['days_since_start'] = (df['date'] - df['date'].min()).dt.days
        
        if len(df) >= 2:
            features_df['hss_trend'] = df['raw_score'].diff().fillna(0)
            features_df['temp_trend'] = df['temp'].diff().fillna(0)
    else:
        features_df['hss_rolling_3'] = df['raw_score']
        features_df['temp_rolling_3'] = df['temp']
        features_df['days_since_start'] = 0
        features_df['hss_trend'] = 0
        features_df['temp_trend'] = 0
    
    # Performance stress indicators
    features_df['environmental_stress'] = (
        normalize(df['temp'], 60, 100) * 0.6 + 
        normalize(df['humidity'], 40, 95) * 0.4
    )
    
    features_df['performance_stress'] = (
        features_df['pace_vs_pr'] * 0.4 + 
        features_df['hr_efficiency'] * 0.6
    )
    
    return features_df

def calculate_heat_index(temp_f, humidity_pct):
    """Calculate heat index (feels like temperature) - vectorized for pandas Series"""
    T = temp_f
    H = humidity_pct
    
    if hasattr(T, '__iter__') and not isinstance(T, str):
        heat_index = np.where(
            T < 80,
            T,
            (
                -42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H
                - 6.83783e-3*T**2 - 5.481717e-2*H**2 + 1.22874e-3*T**2*H
                + 8.5282e-4*T*H**2 - 1.99e-6*T**2*H**2
            )
        )
    else:
        if T < 80:
            heat_index = T
        else:
            heat_index = (
                -42.379 + 2.04901523*T + 10.14333127*H - 0.22475541*T*H
                - 6.83783e-3*T**2 - 5.481717e-2*H**2 + 1.22874e-3*T**2*H
                + 8.5282e-4*T*H**2 - 1.99e-6*T**2*H**2
            )
    
    return heat_index

class HeatAdaptationMLModel:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                max_depth=8, 
                min_samples_split=2, 
                random_state=42
            ),
            'gradient_boost': GradientBoostingRegressor(
                n_estimators=100, 
                max_depth=4, 
                learning_rate=0.1, 
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_names = None
        self.is_trained = False
        
    def prepare_features(self, features_df):
        feature_cols = [
            'temp', 'humidity', 'heat_index', 'pace_vs_pr', 'hr_efficiency',
            'distance', 'temp_humidity_interaction', 'pace_hr_interaction',
            'hss_rolling_3', 'temp_rolling_3', 'days_since_start',
            'environmental_stress', 'performance_stress', 'hss_trend', 'temp_trend'
        ]
        
        X = features_df[feature_cols].fillna(0)
        self.feature_names = feature_cols
        return X
    
    def train_models(self, features_df, target_col='raw_score', min_samples=5):
        if len(features_df) < min_samples:
            st.warning(f"Insufficient data ({len(features_df)} < {min_samples}). Using baseline model.")
            self.is_trained = False
            return None
            
        X = self.prepare_features(features_df)
        y = features_df[target_col]
        
        if len(features_df) < 10:
            X_train, X_test = X, X
            y_train, y_test = y, y
            st.info(f"Using all {len(X)} samples for training (limited data)")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            st.info(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        model_scores = {}
        
        for name, model in self.models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                model_scores[name] = {
                    'model': model,
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'rmse': np.sqrt(mse)
                }
                
        else:
            st.error(f"Error reading CSV file: {str(e)}")
    
    # Analysis section - only show if we have data
    if len(st.session_state.run_data) >= 2:
        st.header("🤖 ML-Enhanced Heat Adaptation Analysis")
        
        if st.button("Run Analysis", type="primary", help="Analyze your heat adaptation potential"):
            run_analysis(st.session_state.run_data, mile_pr_str, max_hr_global)
    
    elif len(st.session_state.run_data) == 1:
        st.info("📊 Add at least one more run to enable analysis.")
    
    else:
        st.info("📊 Add some running data to get started with your heat adaptation analysis!")
    
    # Sample CSV download
    st.sidebar.subheader("📄 Sample Data")
    if st.sidebar.button("Download Sample CSV"):
        sample_data = {
            'date': ['2024-07-01', '2024-07-03', '2024-07-05', '2024-07-07', '2024-07-10'],
            'temp': [85, 88, 92, 89, 91],
            'humidity': [75, 80, 85, 78, 82],
            'pace': ['7:30', '7:45', '8:00', '7:35', '7:50'],
            'avg_hr': [165, 170, 175, 168, 172],
            'distance': [3.1, 5.0, 3.1, 4.0, 6.0]
        }
        sample_df = pd.DataFrame(sample_data)
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        st.sidebar.download_button(
            label="📥 Download sample_heat_data.csv",
            data=csv_buffer.getvalue(),
            file_name="sample_heat_data.csv",
            mime="text/csv"
        )

def run_analysis(run_data, mile_pr_str, max_hr_global):
    """Run the complete heat adaptation analysis"""
    
    with st.spinner("🔄 Running ML-Enhanced Analysis..."):
        # Convert to DataFrame
        df = pd.DataFrame(run_data)
        df_features = create_ml_features(df)
        
        st.success(f"✅ Created {len(df_features.columns)} features for ML analysis")
        
        # Calculate threshold and outliers
        raw_scores = [run['raw_score'] for run in run_data]
        outliers = detect_outliers(raw_scores, method='iqr', threshold=1.5)
        clean_scores = [score for i, score in enumerate(raw_scores) if not outliers[i]]
        
        if len(clean_scores) > 0:
            threshold = estimate_threshold(clean_scores)
            if sum(outliers) > 0:
                outlier_dates = [run_data[i]['date'].strftime('%Y-%m-%d') for i in range(len(outliers)) if outliers[i]]
                st.warning(f"🔍 {sum(outliers)} outlier(s) detected and excluded: {', '.join(outlier_dates)}")
        else:
            threshold = estimate_threshold(raw_scores)
            st.warning("⚠️ All runs detected as outliers. Using all data for threshold.")
        
        # Calculate adjusted and relative scores
        mile_pr_sec = pace_to_seconds(mile_pr_str)
        for run in run_data:
            multiplier = adjust_multiplier(run['raw_score'], threshold, run['pace_sec'], mile_pr_sec)
            adjusted_hss = heat_score(run['temp'], run['humidity'], run['pace_sec'], run['avg_hr'], run['max_hr'], run['distance'], multiplier)
            relative_score = adjusted_hss / threshold if threshold != 0 else 0
            
            run['adjusted_score'] = adjusted_hss
            run['relative_score'] = relative_score
        
        # Update DataFrame
        df['adjusted_score'] = [run['adjusted_score'] for run in run_data]
        df['relative_score'] = [run['relative_score'] for run in run_data]
        df_features['adjusted_score'] = df['adjusted_score']
        
        # Train ML models
        ml_model = HeatAdaptationMLModel()
        clean_df = df_features[~pd.Series(outliers)].copy()
        
        st.subheader("🤖 ML Model Training")
        model_performance = ml_model.train_models(clean_df, target_col='raw_score')
        
        # Prepare arrays for analysis
        dates = np.array([run['date'] for run in run_data])
        raw_scores_arr = np.array([run['raw_score'] for run in run_data])
        adjusted_scores = np.array([run['adjusted_score'] for run in run_data])
        
        # Sort by date
        sort_idx = np.argsort(dates)
        dates = dates[sort_idx]
        raw_scores_arr = raw_scores_arr[sort_idx]
        adjusted_scores = adjusted_scores[sort_idx]
        
        days_since_start = np.array([(d - dates[0]).days for d in dates])
        
        # Model selection and prediction
        MIN_DATA_POINTS_ML = 10
        MIN_DATA_POINTS_COMPLEX = 30
        MIN_DATA_POINTS_SIMPLE = 5
        
        improvement_pct = None
        adapted_same_runs = []
        model_used = "insufficient_data"
        residuals = np.array([0.5, -0.5])
        plateau_days = 8
        
        clean_days = [days_since_start[i] for i in range(len(outliers)) if not outliers[i]]
        clean_adjusted_scores = [adjusted_scores[i] for i in range(len(outliers)) if not outliers[i]]
        
        st.subheader("🎯 Model Selection & Prediction")
        st.info(f"Clean data points available: {len(clean_days)}")
        
        if len(clean_days) >= MIN_DATA_POINTS_ML and ml_model.is_trained:
            model_used = "machine_learning"
            st.success(f"🤖 Using Machine Learning Model ({len(clean_days)} clean data points)")
            
            recent_runs = run_data[-3:] if len(run_data) >= 3 else run_data
            recent_features = df_features.tail(len(recent_runs))
            
            ml_predictions = ml_model.predict(recent_features)
            if ml_predictions is not None:
                recent_val = np.mean(ml_predictions)
            else:
                recent_val = np.mean([run['raw_score'] for run in recent_runs])
            
            # Environmental analysis
            avg_temp = np.mean([run['temp'] for run in recent_runs])
            avg_humidity = np.mean([run['humidity'] for run in recent_runs])
            
            if avg_temp >= 88 and avg_humidity >= 85:
                plateau_days = 12
                base_improvement = 18
            elif avg_temp >= 82 and avg_humidity >= 75:
                plateau_days = 10
                base_improvement = 15
            elif avg_temp >= 75 and avg_humidity >= 65:
                plateau_days = 8
                base_improvement = 12
            else:
                plateau_days = 6
                base_improvement = 8
            
            if model_performance and 'r2' in model_performance:
                if model_performance['r2'] > 0.5:
                    adaptation_factor = min(recent_val / threshold, 2.0) if threshold > 0 else 1.0
                    base_improvement *= adaptation_factor
            
            raw_improvement_pct = base_improvement
            improvement_pct = apply_physiological_limits(raw_improvement_pct, max_improvement=25.0)
            
            if model_performance:
                ml_pred_all = ml_model.predict(clean_df)
                if ml_pred_all is not None:
                    residuals = clean_df['raw_score'].values - ml_pred_all
        
        elif len(clean_days) >= MIN_DATA_POINTS_COMPLEX:
            model_used = "complex_logarithmic"
            st.info(f"📈 Using Complex Logarithmic Model ({len(clean_days)} clean data points)")
            
            Xln = np.log(np.array(clean_days) + 1).reshape(-1, 1)
            X_design = np.hstack([Xln, np.ones_like(Xln)])
            y_fit = np.array(clean_adjusted_scores).reshape(-1, 1)
            
            coef, residuals_sum, _, _ = np.linalg.lstsq(X_design, y_fit, rcond=None)
            a, b = coef.flatten()
            
            fitted_values = (a * np.log(np.array(clean_days) + 1) + b).flatten()
            residuals = np.array(clean_adjusted_scores) - fitted_values
            
            plateau_days = 14
            recent_val = np.mean([score for i, score in enumerate(adjusted_scores) if not outliers[i]][-3:])
            future_mean = a * np.log(clean_days[-1] + plateau_days + 1) + b
            raw_improvement_pct = ((recent_val - future_mean) / recent_val) * 100 if recent_val != 0 else None
            improvement_pct = apply_physiological_limits(raw_improvement_pct, max_improvement=25.0)
        
        elif len(clean_days) >= MIN_DATA_POINTS_SIMPLE:
            model_used = "research_based"
            st.info(f"📚 Using Research-Based Model ({len(clean_days)} clean data points)")
            
            recent_scores = [score for i, score in enumerate(adjusted_scores) if not outliers[i]][-3:]
            baseline_hss = np.mean(recent_scores)
            
            if baseline_hss > 15:
                base_improvement = 20
                plateau_days = 12
            elif baseline_hss > 10:
                base_improvement = 15
                plateau_days = 10
            elif baseline_hss > 6:
                base_improvement = 10
                plateau_days = 8
            else:
                base_improvement = 5
                plateau_days = 6
            
            improvement_pct = apply_physiological_limits(base_improvement, max_improvement=25.0)
        
        else:
            st.warning(f"⚠️ Only {len(clean_days)} clean data points. Need {MIN_DATA_POINTS_SIMPLE}+ for prediction.")
            return
        
        # Calculate adapted performance
        if improvement_pct is not None:
            st.subheader("🔮 Future Adapted Performance Calculation")
            
            # Future dates
            time_gaps = np.diff(dates) if len(dates) > 1 else [timedelta(days=1)]
            adapted_dates = []
            
            start_date = dates[-1] + timedelta(days=plateau_days)
            adapted_dates.append(start_date)
            
            for i in range(len(time_gaps)):
                next_date = adapted_dates[-1] + time_gaps[i]
                adapted_dates.append(next_date)
            
            adapted_dates = np.array(adapted_dates)
            
            # Calculate adapted runs
            adapted_same_runs = []
            for run in run_data:
                adapted_score = run['raw_score'] * (1 - improvement_pct / 100)
                adapted_same_runs.append(max(0, adapted_score))
            
            adapted_same_runs = np.array(adapted_same_runs)
            
            # Confidence intervals
            ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper, ci_95_lower, ci_95_upper = calculate_confidence_intervals(
                adapted_same_runs, residuals, confidence_level=0.95
            )
            
            # Display results
            st.subheader("🎯 Heat Adaptation Predictions")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Used", model_used.replace('_', ' ').title())
            with col2:
                st.metric("Expected Improvement", f"{improvement_pct:.1f}%")
            with col3:
                st.metric("Days to Plateau", f"{plateau_days} days")
            
            # Detailed predictions table
            if len(adapted_same_runs) > 0:
                st.subheader("📊 Run-by-Run Predictions")
                
                predictions_data = []
                for i, (original, adapted) in enumerate(zip([run['raw_score'] for run in run_data], adapted_same_runs)):
                    date_str = run_data[i]['date'].strftime('%Y-%m-%d')
                    improvement = ((original - adapted) / original) * 100
                    outlier_flag = "⚠️" if outliers[i] else "✅"
                    ci_50_range = f"{ci_50_lower[i]:.2f}-{ci_50_upper[i]:.2f}"
                    
                    predictions_data.append({
                        'Status': outlier_flag,
                        'Date': date_str,
                        'Current HSS': f"{original:.2f}",
                        'Adapted HSS': f"{adapted:.2f}",
                        '50% CI Range': ci_50_range,
                        'Improvement': f"{improvement:.1f}%"
                    })
                
                predictions_df = pd.DataFrame(predictions_data)
                st.dataframe(predictions_df, use_container_width=True)
            
            # Generate advice
            recent_scores = [score for i, score in enumerate(adjusted_scores) if not outliers[i]][-3:]
            baseline_hss = np.mean(recent_scores) if recent_scores else np.mean(adjusted_scores)
            generate_adaptation_advice(improvement_pct, baseline_hss, plateau_days, run_data)
            
            # Create visualization
            st.subheader("📈 Heat Adaptation Visualization")
            
            fig = create_visualization(run_data, dates, raw_scores_arr, adjusted_scores, 
                                    adapted_same_runs, adapted_dates, outliers, 
                                    improvement_pct, threshold, model_used, 
                                    ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper, 
                                    ci_95_lower, ci_95_upper, plateau_days)
            
            st.pyplot(fig, use_container_width=True)
            
            # Export results
            st.subheader("📥 Export Results")
            
            # Create summary report
            summary_data = {
                'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M')],
                'Model Used': [model_used.replace('_', ' ').title()],
                'Total Runs': [len(run_data)],
                'Outliers Detected': [sum(outliers)],
                'Expected Improvement': [f"{improvement_pct:.1f}%"],
                'Days to Plateau': [plateau_days],
                'Current Threshold': [f"{threshold:.2f}"],
                'Baseline HSS': [f"{baseline_hss:.2f}"]
            }
            
            summary_df = pd.DataFrame(summary_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv_buffer = io.StringIO()
                summary_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📊 Download Analysis Summary",
                    data=csv_buffer.getvalue(),
                    file_name=f"heat_analysis_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            with col2:
                predictions_csv = io.StringIO()
                predictions_df.to_csv(predictions_csv, index=False)
                st.download_button(
                    label="📈 Download Predictions",
                    data=predictions_csv.getvalue(),
                    file_name=f"heat_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# About section in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ About")
st.sidebar.markdown("""
**Heat Adaptation Analysis v2.0**

This tool uses machine learning to analyze your running performance in heat and predict adaptation improvements.

**Features:**
- ML-enhanced predictions
- Confidence intervals
- Outlier detection
- Personalized advice
- CSV import/export

**Data Requirements:**
- Date, Temperature, Humidity
- Pace, Heart Rate, Distance
- Minimum 2 runs for analysis
""")

if __name__ == "__main__":
    main()
                st.error(f"{name} training failed: {str(e)}")
        
        if model_scores:
            best_name = min(model_scores.keys(), key=lambda k: model_scores[k]['rmse'])
            self.best_model = model_scores[best_name]['model']
            self.is_trained = True
            
            if hasattr(self.best_model, 'feature_importances_'):
                self.display_feature_importance()
                
            return model_scores[best_name]
        else:
            self.is_trained = False
            return None
    
    def predict(self, features_df):
        if not self.is_trained:
            return None
            
        X = self.prepare_features(features_df)
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def display_feature_importance(self):
        if hasattr(self.best_model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            st.subheader("📈 Top Feature Importance")
            for i, row in importance_df.head(5).iterrows():
                st.write(f"**{row['feature']}**: {row['importance']:.3f}")

def estimate_threshold(scores):
    return np.median(scores)

def adjust_multiplier(raw_score, threshold, pace_sec, mile_pr_sec):
    multiplier = 1.0
    if raw_score > threshold:
        multiplier *= 0.9
    if pace_sec <= mile_pr_sec * 1.1:
        multiplier *= 1.1
    return multiplier

def detect_outliers(scores, method='iqr', threshold=1.5):
    scores_array = np.array(scores)
    
    if method == 'iqr':
        Q1 = np.percentile(scores_array, 25)
        Q3 = np.percentile(scores_array, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = (scores_array < lower_bound) | (scores_array > upper_bound)
    
    elif method == 'zscore':
        mean_score = np.mean(scores_array)
        std_score = np.std(scores_array)
        z_scores = np.abs((scores_array - mean_score) / std_score)
        outliers = z_scores > threshold
    
    return outliers

def calculate_confidence_intervals(predictions, residuals, confidence_level=0.95):
    if len(residuals) < 3:
        return predictions, predictions, predictions, predictions, predictions
    
    std_error = np.std(residuals)
    alpha = 1 - confidence_level
    df = len(residuals) - 2
    t_critical = stats.t.ppf(1 - alpha/2, df) if df > 0 else 2.0
    
    t_50 = stats.t.ppf(0.75, df) if df > 0 else 0.67
    t_80 = stats.t.ppf(0.90, df) if df > 0 else 1.28
    t_95 = t_critical
    
    margin_50 = t_50 * std_error
    margin_80 = t_80 * std_error  
    margin_95 = t_95 * std_error
    
    ci_50_lower = predictions - margin_50
    ci_50_upper = predictions + margin_50
    ci_80_lower = predictions - margin_80
    ci_80_upper = predictions + margin_80
    ci_95_lower = predictions - margin_95
    ci_95_upper = predictions + margin_95
    
    return ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper, ci_95_lower, ci_95_upper

def risk_color_adj(score):
    if score < 10:
        return 'green'
    elif score < 15:
        return 'orange'
    else:
        return 'red'

def risk_color_rel(score):
    if score < 0.8:
        return 'green'
    elif score < 1.2:
        return 'orange'
    else:
        return 'red'

def create_visualization(run_data, dates, raw_scores_arr, adjusted_scores, 
                        adapted_same_runs, adapted_dates, outliers, 
                        improvement_pct, threshold, model_used, 
                        ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper, 
                        ci_95_lower, ci_95_upper, plateau_days):
    
    adjusted_colors = [risk_color_adj(s) for s in adjusted_scores]
    relative_scores_list = [run['relative_score'] for run in run_data]
    relative_colors = [risk_color_rel(s) for s in relative_scores_list]

    plt.figure(figsize=(14, 8))

    current_colors = ['red' if outlier else 'blue' for outlier in outliers]
    current_markers = ['x' if outlier else 'o' for outlier in outliers]
    current_sizes = [80 if outlier else 60 for outlier in outliers]

    for i, (date, score, color, marker, size) in enumerate(zip(dates, [run['raw_score'] for run in run_data], current_colors, current_markers, current_sizes)):
        plt.scatter(date, score, color=color, marker=marker, s=size, alpha=0.8, zorder=5)

    plt.plot(dates, [run['raw_score'] for run in run_data], linestyle='-', color='blue', label='Current Raw HSS', linewidth=2, alpha=0.7)
    plt.plot(dates, adjusted_scores, marker='x', linestyle='--', color='purple', label='Adjusted HSS', linewidth=1)
    plt.scatter(dates, adjusted_scores, color=adjusted_colors, s=100, alpha=0.7, label='Adjusted HSS Risk')
    plt.scatter(dates, relative_scores_list, color=relative_colors, s=100, alpha=0.5, edgecolors='k', marker='o', label='Relative HSS Risk')

    if len(adapted_same_runs) > 0:
        plt.plot(adapted_dates, adapted_same_runs, marker='o', linestyle='-', color='darkgreen', 
                 label=f'Same Runs After {plateau_days}d Adaptation ({improvement_pct:.1f}% easier)', linewidth=2, markersize=6, zorder=4)
        
        if len(ci_95_lower) > 0:
            plt.fill_between(adapted_dates, ci_95_lower, ci_95_upper, color='red', alpha=0.15, label='95% CI (Possible)', zorder=1)
            plt.fill_between(adapted_dates, ci_80_lower, ci_80_upper, color='orange', alpha=0.25, label='80% CI (Probable)', zorder=2)
            plt.fill_between(adapted_dates, ci_50_lower, ci_50_upper, color='darkgreen', alpha=0.35, label='50% CI (Most Likely)', zorder=3)

    if sum(outliers) > 0:
        plt.scatter([], [], color='red', marker='x', s=80, label='Outlier Runs (Excluded from Model)', alpha=0.8)

    plt.title(f"Heat Strain Score: Current vs Adapted Performance ({model_used.replace('_', ' ').title()} Model)")
    plt.xlabel("Date")
    plt.ylabel("Heat Strain Score")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    time_span = (max(dates) - min(dates)).days
    if time_span > 30:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    else:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())

    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    
    return plt

def generate_adaptation_advice(improvement_pct, baseline_hss, plateau_days, run_data):
    """Generate personalized heat adaptation training advice"""
    
    # Determine adaptation category
    if improvement_pct >= 18:
        category = "Heat-Naive (High Adaptation Potential)"
    elif improvement_pct >= 12:
        category = "Moderately Heat-Adapted"
    elif improvement_pct >= 8:
        category = "Somewhat Heat-Adapted"
    else:
        category = "Well Heat-Adapted"
    
    st.subheader("🔥 Heat Adaptation Training Recommendations")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-container"><h3>Adaptation Status</h3><p>{category}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><h3>Improvement Potential</h3><p>{improvement_pct:.1f}%</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><h3>Time to Plateau</h3><p>{plateau_days} days</p></div>', unsafe_allow_html=True)
    
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("**⚠️ UNIVERSAL PRECAUTIONS:**")
    st.markdown("""
    • Start conservatively and progress gradually
    • Stop if you experience dizziness, nausea, or excessive fatigue
    • Increase fluid intake starting 2-3 days before heat exposure
    • Consider electrolyte replacement during/after heat sessions
    • Allow 48-72 hours between intense heat adaptation sessions
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("**🌡️ HEAT ADAPTATION TIMELINE:**")
    st.markdown("""
    • **Days 1-3:** Initial physiological responses begin
    • **Days 4-7:** Plasma volume expansion (easier sweating)
    • **Days 8-12:** Improved sweat rate and cooling efficiency
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("**💡 PROGRESS MONITORING:**")
    st.markdown("""
    • Track RPE (Rate of Perceived Exertion) during heat sessions
    • Monitor heart rate response to same-intensity heat exercise
    • Watch for improved comfort in hot conditions
    • Re-run this analysis after {} days to verify improvement
    """.format(plateau_days))
    
    return {'category': category}

# --- STREAMLIT APP MAIN INTERFACE ---

def main():
    st.markdown('<h1 class="main-header">🔥 Heat Adaptation Analysis v2.0</h1>', unsafe_allow_html=True)
    st.markdown("**ML-Enhanced Running Performance in Heat Analysis**")
    
    # Sidebar for inputs
    st.sidebar.header("📝 Setup & Configuration")
    
    # Initial setup
    mile_pr_str = st.sidebar.text_input("Mile PR Pace (MM:SS):", value="7:30", help="Enter your mile personal record pace")
    max_hr_global = st.sidebar.number_input("Max Heart Rate:", min_value=120, max_value=220, value=190, help="Your maximum heart rate")
    
    # Data input method
    st.sidebar.subheader("📊 Data Input Method")
    input_method = st.sidebar.radio("Choose input method:", ["Manual Entry", "CSV Upload"])
    
    # Initialize session state
    if 'run_data' not in st.session_state:
        st.session_state.run_data = []
    
    # Manual entry section
    if input_method == "Manual Entry":
        st.header("📊 Manual Data Entry")
        
        with st.expander("Add New Run", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                date_input = st.date_input("Date of run:", value=datetime.now().date())
                temp = st.number_input("Temperature (°F):", min_value=32, max_value=120, value=85)
                humidity = st.number_input("Humidity (%):", min_value=0, max_value=100, value=75)
            
            with col2:
                pace_str = st.text_input("Pace (MM:SS per mile):", value="8:00")
                avg_hr = st.number_input("Average HR:", min_value=80, max_value=220, value=165)
                distance = st.number_input("Distance (miles):", min_value=0.1, max_value=50.0, value=3.1, step=0.1)
            
            if st.button("Add Run", type="primary"):
                try:
                    date_obj = datetime.combine(date_input, datetime.min.time())
                    pace_sec = pace_to_seconds(pace_str)
                    mile_pr_sec = pace_to_seconds(mile_pr_str)
                    
                    raw_score = heat_score(temp, humidity, pace_sec, avg_hr, max_hr_global, distance, multiplier=1.0)
                    
                    new_run = {
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
                    }
                    
                    st.session_state.run_data.append(new_run)
                    st.success(f"Run added successfully! Heat Strain Score: {raw_score:.2f}")
                    
                except Exception as e:
                    st.error(f"Error adding run: {str(e)}")
        
        # Display current runs
        if st.session_state.run_data:
            st.subheader(f"📋 Current Runs ({len(st.session_state.run_data)} total)")
            
            display_data = []
            for i, run in enumerate(st.session_state.run_data):
                display_data.append({
                    'Date': run['date'].strftime('%Y-%m-%d'),
                    'Temp (°F)': run['temp'],
                    'Humidity (%)': run['humidity'],
                    'Pace': f"{run['pace_sec']//60}:{run['pace_sec']%60:02d}",
                    'Avg HR': run['avg_hr'],
                    'Distance': run['distance'],
                    'HSS': f"{run['raw_score']:.2f}"
                })
            
            df_display = pd.DataFrame(display_data)
            st.dataframe(df_display, use_container_width=True)
            
            if st.button("Clear All Runs", type="secondary"):
                st.session_state.run_data = []
                st.rerun()
    
    # CSV upload section
    elif input_method == "CSV Upload":
        st.header("📁 CSV File Upload")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Upload a CSV with your running data")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
                
                st.subheader("📊 Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
                # Column mapping
                st.subheader("🔍 Column Mapping")
                st.info("Map your CSV columns to the required fields:")
                
                col1, col2 = st.columns(2)
                with col1:
                    date_col = st.selectbox("Date column:", [''] + list(df.columns))
                    temp_col = st.selectbox("Temperature column:", [''] + list(df.columns))
                    humidity_col = st.selectbox("Humidity column:", [''] + list(df.columns))
                
                with col2:
                    pace_col = st.selectbox("Pace column:", [''] + list(df.columns))
                    hr_col = st.selectbox("Heart Rate column:", [''] + list(df.columns))
                    distance_col = st.selectbox("Distance column (optional):", [''] + list(df.columns))
                
                if st.button("Process CSV Data", type="primary"):
                    if all([date_col, temp_col, humidity_col, pace_col, hr_col]):
                        try:
                            run_data = []
                            mile_pr_sec = pace_to_seconds(mile_pr_str)
                            
                            for idx, row in df.iterrows():
                                # Process each row
                                date_str = str(row[date_col])
                                for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%m-%d-%Y', '%d/%m/%Y']:
                                    try:
                                        date_obj = datetime.strptime(date_str, date_format)
                                        break
                                    except ValueError:
                                        continue
                                else:
                                    continue  # Skip if date can't be parsed
                                
                                temp = float(row[temp_col])
                                humidity = float(row[humidity_col])
                                
                                pace_str_csv = str(row[pace_col])
                                if ':' in pace_str_csv:
                                    pace_sec = pace_to_seconds(pace_str_csv)
                                else:
                                    pace_min = float(pace_str_csv)
                                    pace_sec = int(pace_min * 60)
                                
                                avg_hr = float(row[hr_col])
                                distance = float(row[distance_col]) if distance_col else 3.1
                                
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
                            
                            st.session_state.run_data = run_data
                            st.success(f"✅ Successfully processed {len(run_data)} runs!")
                            
                        except Exception as e:
                            st.error(f"Error processing CSV: {str(e)}")
                    else:
                        st.error("Please map all required columns (Date, Temperature, Humidity, Pace, Heart Rate)")
            
            except Exception as e: