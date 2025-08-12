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

# --- FIXED FUNCTIONS ---

def calculate_adaptation_potential_fixed(run_data_list, features_df, ml_model, clean_run_data_list):
    """
    COMPLETELY REWRITTEN: Calculate heat adaptation improvement potential with proper logic
    """
    try:
        # Convert to DataFrame if needed
        if isinstance(clean_run_data_list, list):
            clean_run_data_df = pd.DataFrame(clean_run_data_list)
        else:
            clean_run_data_df = clean_run_data_list
            
        baseline_hss = clean_run_data_df['raw_score'].mean()
        
        # FIXED Factor 1: Performance Level Assessment
        # Elite runners (3:51 mile = 5:58/mile pace) vs recreational runners (6:00 mile pace)
        mile_pr_sec = clean_run_data_df['mile_pr_sec'].iloc[0] if len(clean_run_data_df) > 0 else 360
        mile_pr_pace = mile_pr_sec / 60  # Convert to minutes per mile
        
        if mile_pr_pace <= 5.0:  # Sub-5 minute mile (elite)
            performance_level = "elite"
            base_adaptation_factor = 0.15  # Already highly adapted
        elif mile_pr_pace <= 5.5:  # 5:00-5:30 mile (very competitive)
            performance_level = "very_competitive" 
            base_adaptation_factor = 0.25
        elif mile_pr_pace <= 6.0:  # 5:30-6:00 mile (competitive)
            performance_level = "competitive"
            base_adaptation_factor = 0.35
        elif mile_pr_pace <= 7.0:  # 6:00-7:00 mile (good)
            performance_level = "good"
            base_adaptation_factor = 0.50
        elif mile_pr_pace <= 8.0:  # 7:00-8:00 mile (average)
            performance_level = "average"
            base_adaptation_factor = 0.65
        else:  # >8:00 mile (recreational)
            performance_level = "recreational"
            base_adaptation_factor = 0.80
        
        # FIXED Factor 2: Heat Stress Response Analysis
        # Look at how much the runner slows down in heat relative to their ability
        avg_pace_sec = clean_run_data_df['pace_sec'].mean()
        pace_slowdown_ratio = avg_pace_sec / mile_pr_sec
        
        # Elite runners should maintain closer to PR pace even in heat
        if performance_level == "elite":
            if pace_slowdown_ratio > 1.3:  # Slowing down >30% = not heat adapted
                heat_response_factor = 0.8
            elif pace_slowdown_ratio > 1.15:  # Slowing down 15-30% = somewhat adapted
                heat_response_factor = 0.4
            else:  # Slowing down <15% = well adapted
                heat_response_factor = 0.1
        else:  # Recreational runners
            if pace_slowdown_ratio > 1.5:  # Slowing down >50% = not heat adapted
                heat_response_factor = 0.9
            elif pace_slowdown_ratio > 1.2:  # Slowing down 20-50% = somewhat adapted
                heat_response_factor = 0.6
            else:  # Slowing down <20% = well adapted
                heat_response_factor = 0.3
        
        # FIXED Factor 3: Heart Rate Efficiency in Heat
        avg_hr_efficiency = 0.8  # Default
        hr_efficiencies = []
        
        for _, run in clean_run_data_df.iterrows():
            if (pd.notna(run.get('avg_hr')) and pd.notna(run.get('max_hr')) and run['max_hr'] > 0):
                hr_pct = run['avg_hr'] / run['max_hr']
                # Compare HR% to pace slowdown - inefficient if high HR but slow pace
                pace_ratio = run['pace_sec'] / run['mile_pr_sec']
                
                # Calculate efficiency: ideally HR% should match pace difficulty
                expected_hr_pct = 0.6 + (pace_ratio - 1.0) * 0.3  # Rough estimate
                hr_efficiency = expected_hr_pct / max(hr_pct, 0.5)  # Avoid division by very small numbers
                hr_efficiencies.append(hr_efficiency)
        
        if len(hr_efficiencies) > 0:
            avg_hr_efficiency = np.mean(hr_efficiencies)
            # Lower efficiency = more adaptation potential
            hr_factor = max(0.1, min(1.0, 1.0 - avg_hr_efficiency + 0.3))
        else:
            hr_factor = 0.5
        
        # FIXED Factor 4: Environmental Challenge Level
        temps = clean_run_data_df['temp'].dropna().values
        humidity_vals = clean_run_data_df['humidity'].dropna().values
        
        if len(temps) > 0:
            avg_temp = np.mean(temps)
            avg_humidity = np.mean(humidity_vals) if len(humidity_vals) > 0 else 70
            
            # Calculate heat index for average conditions
            heat_index = calculate_heat_index(avg_temp, avg_humidity)
            
            # Determine environmental challenge
            if heat_index >= 105:  # Extreme heat
                env_challenge = 1.0
            elif heat_index >= 95:   # High heat
                env_challenge = 0.8
            elif heat_index >= 85:   # Moderate heat
                env_challenge = 0.6
            elif heat_index >= 75:   # Mild heat
                env_challenge = 0.4
            else:  # Cool conditions
                env_challenge = 0.2
        else:
            avg_temp = 80
            avg_humidity = 70
            env_challenge = 0.6
        
        # FIXED Factor 5: Consistency Analysis
        # More variable performance = less adapted
        if len(clean_run_data_df) > 1:
            hss_cv = np.std(clean_run_data_df['raw_score']) / np.mean(clean_run_data_df['raw_score'])
            pace_cv = np.std(clean_run_data_df['pace_sec']) / np.mean(clean_run_data_df['pace_sec'])
            
            # High variability = less adapted = more potential
            consistency_factor = min(1.0, (hss_cv + pace_cv) * 2)
        else:
            consistency_factor = 0.5
        
        # COMPLETELY REVISED: Combine factors with proper weighting
        # Elite athletes already have most adaptations, recreational have more potential
        combined_score = (
            base_adaptation_factor * 0.40 +      # Base level (most important)
            heat_response_factor * 0.25 +        # Heat performance
            hr_factor * 0.20 +                   # HR efficiency
            (env_challenge * 0.1) +              # Environmental exposure (small boost for harsh conditions)
            consistency_factor * 0.05            # Performance consistency
        )
        
        # FIXED: Scale improvement percentage based on performance level
        if performance_level == "elite":
            # Elite runners: 2-8% max improvement (already highly adapted)
            improvement_pct = 2 + (combined_score * 6)
        elif performance_level in ["very_competitive", "competitive"]:
            # Good runners: 5-15% improvement
            improvement_pct = 5 + (combined_score * 10)
        elif performance_level == "good":
            # Average competitive: 8-18% improvement
            improvement_pct = 8 + (combined_score * 10)
        else:
            # Recreational runners: 10-25% improvement (most potential)
            improvement_pct = 10 + (combined_score * 15)
        
        # ADDITIONAL LOGIC: Baseline HSS adjustment
        # Very low HSS already indicates good heat adaptation
        if baseline_hss < 5:
            improvement_pct *= 0.5  # Cut in half for already low stress
        elif baseline_hss < 8:
            improvement_pct *= 0.7  # Reduce for moderately low stress
        elif baseline_hss > 15:
            improvement_pct *= 1.2  # Boost for high stress (but cap later)
        
        # Final cap at physiological limits
        improvement_pct = min(improvement_pct, 25.0)
        improvement_pct = max(improvement_pct, 2.0)
        
        return improvement_pct, {
            'baseline_hss': baseline_hss,
            'performance_level': performance_level,
            'mile_pr_pace': mile_pr_pace,
            'pace_slowdown_ratio': pace_slowdown_ratio,
            'base_adaptation_factor': base_adaptation_factor,
            'heat_response_factor': heat_response_factor,
            'hr_factor': hr_factor,
            'env_challenge': env_challenge,
            'consistency_factor': consistency_factor,
            'combined_score': combined_score,
            'avg_temp': avg_temp,
            'avg_humidity': avg_humidity,
            'avg_hr_efficiency': avg_hr_efficiency
        }
        
    except Exception as e:
        st.error(f"Advanced analysis failed: {str(e)}. Using performance-based fallback.")
        
        # Fallback based on performance level
        try:
            mile_pr_sec = clean_run_data_list[0].get('mile_pr_sec', 360)
            mile_pr_pace = mile_pr_sec / 60
            
            if mile_pr_pace <= 5.0:
                improvement_pct = 4  # Elite
            elif mile_pr_pace <= 6.0:
                improvement_pct = 8  # Competitive
            elif mile_pr_pace <= 7.0:
                improvement_pct = 12  # Good
            else:
                improvement_pct = 18  # Recreational
                
        except:
            improvement_pct = 10  # Default
        
        return improvement_pct, {
            'baseline_hss': baseline_hss if 'baseline_hss' in locals() else 10,
            'performance_level': 'unknown',
            'mile_pr_pace': mile_pr_pace if 'mile_pr_pace' in locals() else 6.0,
            'combined_score': 0.5,
            'error': str(e)
        }

def apply_physiological_limits_fixed(improvement_pct, max_improvement=25.0, debug_info=None):
    """Apply realistic physiological limits with better feedback"""
    if improvement_pct is None:
        return None
    
    # Cap improvement at physiologically realistic levels
    limited_improvement = min(abs(improvement_pct), max_improvement)
    
    # Enhanced feedback based on performance level
    if debug_info:
        st.subheader("🔍 Enhanced Adaptation Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Runner Profile:**")
            st.write(f"• Performance Level: {debug_info.get('performance_level', 'Unknown')}")
            st.write(f"• Mile PR: {debug_info.get('mile_pr_pace', 0):.2f} min/mile")
            st.write(f"• Baseline HSS: {debug_info.get('baseline_hss', 0):.1f}")
            st.write(f"• Pace Slowdown: {debug_info.get('pace_slowdown_ratio', 1):.2f}x")
        
        with col2:
            st.write("**Adaptation Factors:**")
            st.write(f"• Base Adaptation: {debug_info.get('base_adaptation_factor', 0):.2f}")
            st.write(f"• Heat Response: {debug_info.get('heat_response_factor', 0):.2f}")
            st.write(f"• HR Efficiency: {debug_info.get('hr_factor', 0):.2f}")
            st.write(f"• Environmental: {debug_info.get('env_challenge', 0):.2f}")
        
        # Performance interpretation
        performance_level = debug_info.get('performance_level', 'unknown')
        if performance_level == 'elite':
            st.info("🏃‍♀️ **Elite Runner**: Already highly heat-adapted. Limited improvement potential.")
        elif performance_level in ['very_competitive', 'competitive']:
            st.info("🏃‍♂️ **Competitive Runner**: Some heat adaptation potential remains.")
        elif performance_level == 'good':
            st.info("🏃‍♀️ **Good Runner**: Moderate heat adaptation potential.")
        else:
            st.info("🏃‍♂️ **Recreational Runner**: High heat adaptation potential - lots of room for improvement!")
    
    return limited_improvement

def calculate_adaptation_days_fixed(improvement_pct, run_data, performance_level=None):
    """
    Calculate days to adaptation plateau based on performance level and improvement potential
    """
    # Base days by performance level (elite adapt faster due to training experience)
    if performance_level == "elite":
        base_days = 8   # Elite athletes adapt quickly
    elif performance_level in ["very_competitive", "competitive"]:
        base_days = 10  # Competitive athletes adapt reasonably fast
    elif performance_level == "good":
        base_days = 12  # Good athletes need moderate time
    else:
        base_days = 14  # Recreational athletes need more time
    
    # Adjust based on improvement potential
    if improvement_pct >= 20:
        base_days += 4  # High potential = longer adaptation needed
    elif improvement_pct >= 15:
        base_days += 2  # Moderate potential
    elif improvement_pct <= 5:
        base_days -= 2  # Low potential = already mostly adapted
    
    # Training frequency adjustment
    if len(run_data) > 0:
        if isinstance(run_data, list):
            dates = [run['date'] for run in run_data if 'date' in run]
        else:
            dates = run_data['date'].tolist()
            
        if len(dates) > 1:
            date_range = (max(dates) - min(dates)).days
            if date_range > 0:
                runs_per_week = len(run_data) * 7 / date_range
                if runs_per_week >= 6:
                    base_days -= 2  # High frequency = faster adaptation
                elif runs_per_week >= 4:
                    base_days -= 1  # Regular training
                elif runs_per_week <= 2:
                    base_days += 2  # Low frequency = slower adaptation
    
    return max(7, min(21, base_days))

def generate_adaptation_advice_fixed(improvement_pct, baseline_hss, plateau_days, run_data, debug_info):
    """Generate advice based on actual performance level and adaptation potential"""
    
    performance_level = debug_info.get('performance_level', 'unknown')
    mile_pr_pace = debug_info.get('mile_pr_pace', 6.0)
    
    # Determine category based on performance level AND improvement potential
    if performance_level == "elite":
        if improvement_pct >= 6:
            category = "Elite with Heat Weakness"
            advice_level = "elite_needs_work"
        else:
            category = "Heat-Adapted Elite"
            advice_level = "elite_maintain"
    elif performance_level in ["very_competitive", "competitive"]:
        if improvement_pct >= 12:
            category = "Competitive Heat-Naive"
            advice_level = "competitive_develop"
        else:
            category = "Heat-Adapted Competitive"
            advice_level = "competitive_maintain"
    else:  # good, average, recreational
        if improvement_pct >= 15:
            category = "Heat-Naive Recreational"
            advice_level = "recreational_develop"
        else:
            category = "Heat-Experienced Recreational"
            advice_level = "recreational_maintain"
    
    st.subheader("🔥 Performance-Based Heat Training Plan")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-container"><h3>Runner Category</h3><p>{category}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><h3>Improvement Potential</h3><p>{improvement_pct:.1f}%</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><h3>Adaptation Timeline</h3><p>{plateau_days} days</p></div>', unsafe_allow_html=True)
    
    # Universal precautions
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("**⚠️ SAFETY FIRST:**")
    st.markdown("""
    • Always prioritize safety over performance gains
    • Heat adaptation requires progressive overload - start conservatively
    • Monitor for signs of heat illness: excessive fatigue, dizziness, nausea, confusion
    • Maintain proper hydration before, during, and after heat exposure
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance-specific advice
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    
    if advice_level == "elite_needs_work":
        st.markdown(f"**🏆 ELITE HEAT DEVELOPMENT (PR: {mile_pr_pace:.2f}/mile):**")
        st.markdown("""
        • **Focus:** Technical heat management - you have the fitness, need the adaptation
        • **Protocol:** 2-3 heat sessions/week, maintain training intensity
        • **Key workouts:** Tempo runs and intervals in 85-95°F conditions
        • **Timeline:** 8-12 days for measurable improvement
        • **Expected:** 4-8% performance gain in hot conditions
        """)
    elif advice_level == "elite_maintain":
        st.markdown(f"**🏆 ELITE HEAT MAINTENANCE (PR: {mile_pr_pace:.2f}/mile):**")
        st.markdown("""
        • **Status:** Already heat-adapted at elite level
        • **Protocol:** 1-2 heat sessions/week to maintain adaptations
        • **Focus:** Race-specific heat strategies and fine-tuning
        • **Gains:** 2-4% refinement possible through tactical improvements
        """)
    elif advice_level == "competitive_develop":
        st.markdown(f"**🏃‍♂️ COMPETITIVE HEAT DEVELOPMENT (PR: {mile_pr_pace:.2f}/mile):**")
        st.markdown("""
        • **Week 1-2:** Build heat tolerance with 45-60min easy runs in warm conditions
        • **Week 2-3:** Add structured workouts: tempo runs and short intervals
        • **Progressive exposure:** Start at 80°F, progress to race conditions
        • **Expected:** 10-15% improvement in heat performance
        """)
    elif advice_level == "competitive_maintain":
        st.markdown(f"**🏃‍♂️ COMPETITIVE HEAT MAINTENANCE (PR: {mile_pr_pace:.2f}/mile):**")
        st.markdown("""
        • **Status:** Good heat fitness for your performance level
        • **Protocol:** 2 heat sessions/week during hot seasons
        • **Focus:** Maintain current adaptations, work on heat race tactics
        • **Gains:** 5-10% optimization possible
        """)
    elif advice_level == "recreational_develop":
        st.markdown(f"**🏃‍♀️ RECREATIONAL HEAT DEVELOPMENT (PR: {mile_pr_pace:.2f}/mile):**")
        st.markdown("""
        • **Huge potential!** You can make significant gains in heat performance
        • **Week 1:** Start with 20-30min easy runs in moderate heat (75-80°F)
        • **Week 2-3:** Progress to 45-60min, add one tempo effort per week
        • **Week 3-4:** Build to normal training duration in target conditions
        • **Expected:** 15-25% improvement - this will transform your summer running!
        """)
    else:  # recreational_maintain
        st.markdown(f"**🏃‍♀️ RECREATIONAL HEAT MAINTENANCE (PR: {mile_pr_pace:.2f}/mile):**")
        st.markdown("""
        • **Good work!** You've already developed solid heat fitness
        • **Protocol:** 2-3 heat runs per week to maintain adaptations
        • **Focus:** Enjoy consistent comfort in hot weather
        • **Gains:** 8-12% refinement through improved pacing and fueling
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {'category': category, 'advice_level': advice_level, 'performance_level': performance_level}

# --- SUPPORTING FUNCTIONS ---

def normalize(value, min_val, max_val):
    """Normalize value(s) to 0-1 range, handling both scalars and pandas Series"""
    if max_val == min_val:
        return 0.5 if np.isscalar(value) else np.full_like(value, 0.5, dtype=float)
    
    normalized = (value - min_val) / (max_val - min_val)
    
    if np.isscalar(normalized):
        return max(0, min(1, normalized))
    else:
        # Handle pandas Series or numpy arrays
        return np.clip(normalized, 0, 1)

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
            features_df['hss_trend'] = 0
            features_df['temp_trend'] = 0
    else:
        features_df['hss_rolling_3'] = df['raw_score']
        features_df['temp_rolling_3'] = df['temp']
        features_df['days_since_start'] = 0
        features_df['hss_trend'] = 0
        features_df['temp_trend'] = 0
    
    # Performance stress indicators - Fixed to handle NaN values
    temp_normalized = normalize(df['temp'].fillna(80), 60, 100)
    humidity_normalized = normalize(df['humidity'].fillna(70), 40, 95)
    
    features_df['environmental_stress'] = (
        temp_normalized * 0.6 + 
        humidity_normalized * 0.4
    )
    
    features_df['performance_stress'] = (
        features_df['pace_vs_pr'].fillna(1.0) * 0.4 + 
        features_df['hr_efficiency'].fillna(0.8) * 0.6
    )
    
    # Fill any remaining NaN values
    features_df = features_df.fillna(0)
    
    return features_df

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
                
            except Exception as e:
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
        return predictions, predictions, predictions, predictions, predictions, predictions
    
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

def create_visualization(run_data_df, dates, raw_scores_arr, adjusted_scores, 
                        adapted_same_runs, adapted_dates, outliers, 
                        improvement_pct, threshold, model_used, 
                        ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper, 
                        ci_95_lower, ci_95_upper, plateau_days):
    
    adjusted_colors = [risk_color_adj(s) for s in adjusted_scores]
    relative_scores_list = run_data_df['relative_score'].tolist()
    relative_colors = [risk_color_rel(s) for s in relative_scores_list]

    plt.figure(figsize=(14, 8))

    current_colors = ['red' if outlier else 'blue' for outlier in outliers]
    current_markers = ['x' if outlier else 'o' for outlier in outliers]
    current_sizes = [80 if outlier else 60 for outlier in outliers]

    raw_scores_list = run_data_df['raw_score'].tolist()
    for i, (date, score, color, marker, size) in enumerate(zip(dates, raw_scores_list, current_colors, current_markers, current_sizes)):
        plt.scatter(date, score, color=color, marker=marker, s=size, alpha=0.8, zorder=5)

    plt.plot(dates, raw_scores_list, linestyle='-', color='blue', label='Current Raw HSS', linewidth=2, alpha=0.7)
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

# --- STREAMLIT APP MAIN INTERFACE ---

def main():
    st.markdown('<h1 class="main-header">🔥 Heat Adaptation Analysis v2.0</h1>', unsafe_allow_html=True)
    st.markdown("**ML-Enhanced Running Performance in Heat Analysis with Fixed Adaptation Logic**")
    
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
                        st.warning("Please select all required columns (Date, Temperature, Humidity, Pace, Heart Rate)")
                        
            except Exception as e:
                st.error(f"Error loading CSV file: {str(e)}")
    
    # Analysis section - only show if we have data
    if st.session_state.run_data and len(st.session_state.run_data) >= 2:
        st.header("🔬 Heat Adaptation Analysis with Fixed Logic")
        
        # Analysis configuration
        st.sidebar.subheader("🔧 Analysis Settings")
        outlier_method = st.sidebar.selectbox("Outlier detection method:", ['iqr', 'zscore'])
        outlier_threshold = st.sidebar.slider("Outlier threshold:", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        
        if st.button("🚀 Run Fixed Analysis", type="primary"):
            with st.spinner("Running enhanced ML analysis with fixed adaptation logic..."):
                try:
                    # Prepare data - convert list to DataFrame
                    run_data_list = st.session_state.run_data.copy()
                    df = pd.DataFrame(run_data_list)
                    df = df.sort_values('date').reset_index(drop=True)
                    
                    # Create ML features
                    features_df = create_ml_features(df)
                    
                    # Detect outliers
                    raw_scores = df['raw_score'].tolist()
                    outliers = detect_outliers(raw_scores, method=outlier_method, threshold=outlier_threshold)
                    
                    # Filter out outliers for model training
                    clean_features_df = features_df[~outliers].copy()
                    clean_run_data_df = df[~outliers].copy()
                    clean_run_data_list = [run_data_list[i] for i in range(len(run_data_list)) if not outliers[i]]
                    
                    # Train ML model
                    ml_model = HeatAdaptationMLModel()
                    model_performance = ml_model.train_models(clean_features_df)
                    
                    # Calculate baseline metrics
                    threshold = estimate_threshold(clean_run_data_df['raw_score'].tolist())
                    
                    # Calculate adjusted scores and relative scores
                    adjusted_scores = []
                    relative_scores = []
                    
                    for i, (_, run) in enumerate(df.iterrows()):
                        multiplier = adjust_multiplier(run['raw_score'], threshold, run['pace_sec'], run['mile_pr_sec'])
                        adjusted_score = run['raw_score'] * multiplier
                        relative_score = run['raw_score'] / threshold
                        
                        adjusted_scores.append(adjusted_score)
                        relative_scores.append(relative_score)
                    
                    # Update DataFrame with calculated scores
                    df['adjusted_score'] = adjusted_scores
                    df['relative_score'] = relative_scores
                    
                    # Update the original list as well
                    for i, run in enumerate(run_data_list):
                        run['adjusted_score'] = adjusted_scores[i]
                        run['relative_score'] = relative_scores[i]
                    
                    # ML predictions and adaptation modeling
                    if ml_model.is_trained:
                        predictions = ml_model.predict(features_df)
                        residuals = df['raw_score'].values - predictions
                        model_used = "ML Model"
                    else:
                        # Fallback to statistical model
                        X = np.arange(len(df)).reshape(-1, 1)
                        y = clean_run_data_df['raw_score'].values
                        
                        if len(clean_run_data_df) >= 3:
                            # Use statsmodels for trend analysis
                            X_sm = sm.add_constant(X[:len(clean_run_data_df)])
                            try:
                                model = sm.OLS(y, X_sm).fit()
                                predictions = model.predict(sm.add_constant(X))
                                residuals = df['raw_score'].values - predictions
                                model_used = "Statistical Model"
                            except:
                                predictions = np.full(len(df), np.mean(y))
                                residuals = df['raw_score'].values - predictions
                                model_used = "Simple Average"
                        else:
                            predictions = np.full(len(df), np.mean(y))
                            residuals = df['raw_score'].values - predictions
                            model_used = "Simple Average"
                    
                    # *** USE THE FIXED FUNCTIONS HERE ***
                    # Calculate improvement percentage using the FIXED function
                    improvement_pct, debug_info = calculate_adaptation_potential_fixed(
                        df, features_df, ml_model, clean_run_data_df
                    )
                    
                    # Apply physiological limits using FIXED function
                    improvement_pct = apply_physiological_limits_fixed(improvement_pct, debug_info=debug_info)
                    
                    # Calculate adaptation days using FIXED function
                    plateau_days = calculate_adaptation_days_fixed(improvement_pct, clean_run_data_df, debug_info.get('performance_level'))
                    
                    # Calculate baseline HSS
                    baseline_hss = clean_run_data_df['raw_score'].mean()
                    
                    # Create adapted scenarios
                    adapted_same_runs = []
                    adapted_dates = []
                    
                    for _, run in df.iterrows():
                        adapted_score = run['raw_score'] * (1 - improvement_pct / 100)
                        adapted_same_runs.append(adapted_score)
                        adapted_dates.append(run['date'] + timedelta(days=plateau_days))
                    
                    # Calculate confidence intervals
                    ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper, ci_95_lower, ci_95_upper = calculate_confidence_intervals(
                        np.array(adapted_same_runs), residuals
                    )
                    
                    # Create visualization
                    dates = df['date'].tolist()
                    raw_scores = df['raw_score'].tolist()
                    
                    fig = create_visualization(
                        df, dates, raw_scores, adjusted_scores, 
                        adapted_same_runs, adapted_dates, outliers,
                        improvement_pct, threshold, model_used,
                        ci_50_lower, ci_50_upper, ci_80_lower, ci_80_upper,
                        ci_95_lower, ci_95_upper, plateau_days
                    )
                    
                    # Display results
                    st.subheader("📊 Fixed Analysis Results")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Baseline HSS", f"{baseline_hss:.2f}")
                    with col2:
                        st.metric("Threshold", f"{threshold:.2f}")
                    with col3:
                        st.metric("Improvement", f"{improvement_pct:.1f}%")
                    with col4:
                        st.metric("Outliers", f"{sum(outliers)}/{len(outliers)}")
                    
                    # Show the plot
                    st.pyplot(fig)
                    
                    # Model performance metrics
                    if model_performance:
                        st.subheader("🤖 Model Performance")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("R² Score", f"{model_performance['r2']:.3f}")
                        with col2:
                            st.metric("RMSE", f"{model_performance['rmse']:.3f}")
                        with col3:
                            st.metric("MAE", f"{model_performance['mae']:.3f}")
                    
                    # Generate FIXED training advice
                    advice = generate_adaptation_advice_fixed(improvement_pct, baseline_hss, plateau_days, df, debug_info)
                    
                    # Summary statistics
                    st.subheader("📈 Summary Statistics")
                    summary_df = pd.DataFrame({
                        'Metric': ['Mean HSS', 'Median HSS', 'Std Dev', 'Min HSS', 'Max HSS', 'Range'],
                        'Current': [
                            np.mean(raw_scores),
                            np.median(raw_scores),
                            np.std(raw_scores),
                            np.min(raw_scores),
                            np.max(raw_scores),
                            np.max(raw_scores) - np.min(raw_scores)
                        ],
                        'After Adaptation': [
                            np.mean(adapted_same_runs),
                            np.median(adapted_same_runs),
                            np.std(adapted_same_runs),
                            np.min(adapted_same_runs),
                            np.max(adapted_same_runs),
                            np.max(adapted_same_runs) - np.min(adapted_same_runs)
                        ]
                    })
                    
                    st.dataframe(summary_df, use_container_width=True)
                    
                    # Export options
                    st.subheader("💾 Export Options")
                    
                    # Prepare export data
                    export_df = pd.DataFrame([
                        {
                            'Date': run['date'].strftime('%Y-%m-%d'),
                            'Temperature': run['temp'],
                            'Humidity': run['humidity'],
                            'Pace_Seconds': run['pace_sec'],
                            'Avg_HR': run['avg_hr'],
                            'Distance': run['distance'],
                            'Raw_HSS': run['raw_score'],
                            'Adjusted_HSS': run['adjusted_score'],
                            'Relative_HSS': run['relative_score'],
                            'Outlier': outliers[i],
                            'Adapted_HSS': adapted_same_runs[i],
                            'Performance_Level': debug_info.get('performance_level', 'unknown'),
                            'Mile_PR_Pace': debug_info.get('mile_pr_pace', 0),
                            'Improvement_Pct': improvement_pct,
                            'Plateau_Days': plateau_days
                        } for i, run in enumerate(run_data_list)
                    ])
                    
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Fixed Analysis Results as CSV",
                        data=csv_data,
                        file_name=f"heat_adaptation_fixed_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"Analysis failed: {str(e)}")
                    st.error("Please check your data and try again.")
                    # Show the full traceback for debugging
                    import traceback
                    st.code(traceback.format_exc())
                    
    elif st.session_state.run_data and len(st.session_state.run_data) == 1:
        st.info("Add at least one more run to perform analysis.")
        
    else:
        st.info("Add some running data to get started with the analysis!")
    
    # Footer with information
    st.markdown("---")
    st.markdown("### ℹ️ About This Fixed Analysis")
    with st.expander("Key Fixes Implemented"):
        st.markdown("""
        **🔧 Fixed Adaptation Potential Logic:**
        - **Performance Level Assessment**: Elite runners (sub-5:00 mile) vs recreational runners (8:00+ mile)
        - **Heat Response Analysis**: How much pace slows down relative to performance level
        - **HR Efficiency**: Heart rate response efficiency in heat conditions
        - **Environmental Challenge**: Actual heat index exposure and variability
        - **Consistency Analysis**: Performance variability indicates adaptation status
        
        **📊 Proper Differentiation:**
        - Elite runners: 2-8% improvement potential (already highly adapted)
        - Competitive runners: 5-15% improvement potential
        - Recreational runners: 10-25% improvement potential (most room for growth)
        
        **🎯 Performance-Based Recommendations:**
        - Training advice tailored to actual performance level
        - Realistic timeline expectations based on current fitness
        - Safety-first approach with progressive overload
        """)
    
    with st.expander("Testing the Fixed Logic"):
        st.markdown("""
        **🧪 Test Scenarios to Verify the Fix:**
        
        **Heat-Naive Elite Runner (Should show LOW potential ~4-6%):**
        - Mile PR: 4:50 (elite level)
        - Current pace in heat: 6:30 (much slower than PR)
        - High heart rate despite slow pace
        - Expected: Low improvement % because already elite, just needs heat exposure
        
        **Heat-Naive Recreational Runner (Should show HIGH potential ~18-25%):**
        - Mile PR: 9:00 (recreational level) 
        - Current pace in heat: 11:00+ (very slow)
        - High heart rate for the effort
        - Expected: High improvement % because lots of room for heat adaptation
        
        **Heat-Adapted Elite Runner (Should show VERY LOW potential ~2-4%):**
        - Mile PR: 4:50 (elite level)
        - Current pace in heat: 5:10 (close to PR pace)
        - Efficient heart rate response
        - Expected: Minimal improvement % because already elite AND heat-adapted
        
        **Heat-Adapted Recreational Runner (Should show MODERATE potential ~8-12%):**
        - Mile PR: 9:00 (recreational level)
        - Current pace in heat: 9:30 (reasonable slowdown)
        - Decent heart rate efficiency
        - Expected: Moderate improvement % - some room for refinement
        """)
    
    with st.expander("How it works"):
        st.markdown("""
        **Heat Strain Score (HSS)** quantifies the physiological stress of running in heat based on:
        - Temperature and humidity conditions
        - Your running pace vs. personal record
        - Heart rate response
        - Distance covered
        
        **Machine Learning Models** analyze your data to predict:
        - Heat adaptation potential
        - Performance improvements over time
        - Confidence intervals for predictions
        
        **Key Features:**
        - Outlier detection and filtering
        - Multiple modeling approaches (Random Forest, Gradient Boosting)
        - Physiologically-constrained improvement estimates
        - Performance-level-based training recommendations
        """)
    
    with st.expander("Data Requirements"):
        st.markdown("""
        **Required Data:**
        - Date of each run
        - Temperature (°F)
        - Humidity (%)
        - Running pace (MM:SS format)
        - Average heart rate
        - Mile PR pace (for performance level assessment)
        
        **Optional Data:**
        - Distance (defaults to 3.1 miles if not provided)
        
        **CSV Format:**
        Your CSV should have columns for each required field. The app will help you map column names during upload.
        """)

if __name__ == "__main__":
    main()
