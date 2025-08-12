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

def pace_to_seconds_fixed(pace_str):
    """Convert pace string to seconds - FIXED"""
    if isinstance(pace_str, str) and ':' in pace_str:
        minutes, seconds = map(int, pace_str.split(":"))
        return minutes * 60 + seconds
    else:
        # Handle decimal minutes format - convert to MM:SS first
        pace_min = float(pace_str)
        minutes = int(pace_min)
        seconds = int((pace_min - minutes) * 60)
        return minutes * 60 + seconds

def calculate_adaptation_potential_fixed_v2(run_data_list, features_df, ml_model, clean_run_data_list):
    """
    COMPLETELY REWRITTEN with proper performance level assessment
    """
    try:
        # Convert to DataFrame if needed
        if isinstance(clean_run_data_list, list):
            clean_run_data_df = pd.DataFrame(clean_run_data_list)
        else:
            clean_run_data_df = clean_run_data_list
            
        baseline_hss = clean_run_data_df['raw_score'].mean()
        
        # FIXED: Proper mile PR assessment
        mile_pr_sec = clean_run_data_df['mile_pr_sec'].iloc[0] if len(clean_run_data_df) > 0 else 450
        
        # Convert seconds to MM:SS for display
        pr_minutes = mile_pr_sec // 60
        pr_seconds = mile_pr_sec % 60
        mile_pr_display = f"{pr_minutes}:{pr_seconds:02d}"
        
        print(f"DEBUG: Mile PR = {mile_pr_sec} seconds ({mile_pr_display})")
        
        # CORRECTED Performance Level Assessment
        if mile_pr_sec <= 300:  # Sub-5:00 mile (300 seconds) - Elite
            performance_level = "elite"
            base_adaptation_potential = 0.06  # 6% max for elite
        elif mile_pr_sec <= 330:  # 5:00-5:30 mile - Very competitive
            performance_level = "very_competitive" 
            base_adaptation_potential = 0.10  # 10% max
        elif mile_pr_sec <= 360:  # 5:30-6:00 mile - Competitive
            performance_level = "competitive"
            base_adaptation_potential = 0.15  # 15% max
        elif mile_pr_sec <= 420:  # 6:00-7:00 mile - Good recreational
            performance_level = "good"
            base_adaptation_potential = 0.18  # 18% max
        elif mile_pr_sec <= 480:  # 7:00-8:00 mile - Average recreational
            performance_level = "average"
            base_adaptation_potential = 0.22  # 22% max
        else:  # 8:00+ mile - Beginner recreational
            performance_level = "recreational"
            base_adaptation_potential = 0.25  # 25% max
        
        print(f"DEBUG: Performance level = {performance_level}, Base potential = {base_adaptation_potential}")
        
        # FIXED: Heat Adaptation Status Assessment
        # Calculate how much they slow down in heat relative to PR
        avg_pace_sec = clean_run_data_df['pace_sec'].mean()
        pace_slowdown_factor = avg_pace_sec / mile_pr_sec
        
        print(f"DEBUG: Avg pace = {avg_pace_sec//60}:{avg_pace_sec%60:02d} ({avg_pace_sec}s)")
        print(f"DEBUG: Slowdown factor = {pace_slowdown_factor:.2f}")
        
        # Determine heat adaptation status based on performance level expectations
        if performance_level == "elite":
            if pace_slowdown_factor <= 1.10:  # <10% slowdown = heat adapted
                heat_adaptation_status = "adapted"
                adaptation_multiplier = 0.3  # Only 30% of potential remains
            elif pace_slowdown_factor <= 1.25:  # 10-25% slowdown = partially adapted
                heat_adaptation_status = "partially_adapted"
                adaptation_multiplier = 0.6  # 60% of potential remains
            else:  # >25% slowdown = heat naive (surprising for elite!)
                heat_adaptation_status = "naive"
                adaptation_multiplier = 1.0  # Full potential available
        elif performance_level in ["very_competitive", "competitive"]:
            if pace_slowdown_factor <= 1.15:  # <15% slowdown = adapted
                heat_adaptation_status = "adapted"
                adaptation_multiplier = 0.4  # 40% of potential remains
            elif pace_slowdown_factor <= 1.35:  # 15-35% slowdown = partially adapted
                heat_adaptation_status = "partially_adapted"
                adaptation_multiplier = 0.7  # 70% of potential remains
            else:  # >35% slowdown = naive
                heat_adaptation_status = "naive"
                adaptation_multiplier = 1.0  # Full potential
        else:  # recreational runners
            if pace_slowdown_factor <= 1.20:  # <20% slowdown = adapted (good for recreational)
                heat_adaptation_status = "adapted"
                adaptation_multiplier = 0.5  # 50% of potential remains
            elif pace_slowdown_factor <= 1.50:  # 20-50% slowdown = partially adapted
                heat_adaptation_status = "partially_adapted"
                adaptation_multiplier = 0.8  # 80% of potential remains
            else:  # >50% slowdown = very heat naive
                heat_adaptation_status = "naive"
                adaptation_multiplier = 1.0  # Full potential
        
        print(f"DEBUG: Heat adaptation status = {heat_adaptation_status}")
        print(f"DEBUG: Adaptation multiplier = {adaptation_multiplier}")
        
        # FIXED: Heart Rate Efficiency (minor adjustment factor)
        hr_efficiency_factor = 1.0  # Default neutral
        
        if not clean_run_data_df['avg_hr'].isna().all() and not clean_run_data_df['max_hr'].isna().all():
            avg_hr_pct = (clean_run_data_df['avg_hr'] / clean_run_data_df['max_hr']).mean()
            
            # For heat naive runners, high HR% suggests more room for improvement
            # For heat adapted runners, efficient HR suggests less room
            if heat_adaptation_status == "naive":
                if avg_hr_pct > 0.85:  # Very high HR in heat = more potential
                    hr_efficiency_factor = 1.1
                elif avg_hr_pct < 0.75:  # Reasonable HR in heat = already somewhat efficient
                    hr_efficiency_factor = 0.9
            else:  # adapted runners
                if avg_hr_pct > 0.80:  # Still high HR despite adaptation = some room left
                    hr_efficiency_factor = 1.05
                elif avg_hr_pct < 0.70:  # Very efficient = little room left
                    hr_efficiency_factor = 0.85
        
        # FIXED: Environmental Factor (small bonus for harsh conditions)
        env_factor = 1.0
        if not clean_run_data_df['temp'].isna().all():
            avg_temp = clean_run_data_df['temp'].mean()
            if avg_temp >= 90:  # Very hot conditions = slight bonus
                env_factor = 1.05
            elif avg_temp >= 85:  # Hot conditions = small bonus
                env_factor = 1.02
        
        # FIXED: Final Calculation
        # Base potential * adaptation status * minor adjustments
        improvement_pct = (
            base_adaptation_potential * 
            adaptation_multiplier * 
            hr_efficiency_factor * 
            env_factor * 
            100  # Convert to percentage
        )
        
        print(f"DEBUG: Final improvement = {base_adaptation_potential} * {adaptation_multiplier} * {hr_efficiency_factor} * {env_factor} * 100 = {improvement_pct:.1f}%")
        
        # REMOVED: No more baseline HSS override that was ruining everything!
        
        # Final bounds check
        if performance_level == "elite":
            improvement_pct = max(1.0, min(improvement_pct, 8.0))
        elif performance_level in ["very_competitive", "competitive"]:
            improvement_pct = max(2.0, min(improvement_pct, 15.0))
        else:
            improvement_pct = max(3.0, min(improvement_pct, 25.0))
        
        return improvement_pct, {
            'baseline_hss': baseline_hss,
            'performance_level': performance_level,
            'mile_pr_display': mile_pr_display,
            'mile_pr_sec': mile_pr_sec,
            'avg_pace_sec': avg_pace_sec,
            'pace_slowdown_factor': pace_slowdown_factor,
            'heat_adaptation_status': heat_adaptation_status,
            'base_adaptation_potential': base_adaptation_potential,
            'adaptation_multiplier': adaptation_multiplier,
            'hr_efficiency_factor': hr_efficiency_factor,
            'env_factor': env_factor,
            'avg_temp': clean_run_data_df['temp'].mean() if not clean_run_data_df['temp'].isna().all() else 80,
        }
        
    except Exception as e:
        print(f"ERROR in adaptation calculation: {str(e)}")
        # Fallback based on performance level only
        try:
            mile_pr_sec = clean_run_data_list[0].get('mile_pr_sec', 450)
            
            if mile_pr_sec <= 300:  # Elite
                improvement_pct = 4
                performance_level = "elite"
            elif mile_pr_sec <= 360:  # Competitive
                improvement_pct = 10
                performance_level = "competitive"
            else:  # Recreational
                improvement_pct = 18
                performance_level = "recreational"
                
        except:
            improvement_pct = 12  # Default
            performance_level = "unknown"
        
        return improvement_pct, {
            'baseline_hss': baseline_hss if 'baseline_hss' in locals() else 10,
            'performance_level': performance_level,
            'mile_pr_display': "Unknown",
            'heat_adaptation_status': 'unknown',
            'error': str(e)
        }

def apply_physiological_limits_fixed_v2(improvement_pct, max_improvement=25.0, debug_info=None):
    """Apply realistic physiological limits with enhanced feedback"""
    if improvement_pct is None:
        return None
    
    limited_improvement = min(abs(improvement_pct), max_improvement)
    
    if debug_info:
        st.subheader("🔍 FIXED Heat Adaptation Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Runner Profile:**")
            st.write(f"• Performance Level: **{debug_info.get('performance_level', 'Unknown').replace('_', ' ').title()}**")
            st.write(f"• Mile PR: **{debug_info.get('mile_pr_display', 'Unknown')}**")
            st.write(f"• Current Avg Pace: **{int(debug_info.get('avg_pace_sec', 0)//60)}:{int(debug_info.get('avg_pace_sec', 0)%60):02d}**")
            st.write(f"• Pace Slowdown: **{debug_info.get('pace_slowdown_factor', 1):.2f}x**")
            st.write(f"• Heat Adaptation Status: **{debug_info.get('heat_adaptation_status', 'unknown').replace('_', ' ').title()}**")
        
        with col2:
            st.write("**Adaptation Analysis:**")
            st.write(f"• Base Potential: **{debug_info.get('base_adaptation_potential', 0)*100:.1f}%**")
            st.write(f"• Adaptation Multiplier: **{debug_info.get('adaptation_multiplier', 0):.2f}**")
            st.write(f"• HR Efficiency Factor: **{debug_info.get('hr_efficiency_factor', 1):.2f}**")
            st.write(f"• Environment Factor: **{debug_info.get('env_factor', 1):.2f}**")
            st.write(f"• **Final Improvement: {limited_improvement:.1f}%**")
        
        # Enhanced interpretation
        performance_level = debug_info.get('performance_level', 'unknown')
        adaptation_status = debug_info.get('heat_adaptation_status', 'unknown')
        
        if performance_level == 'elite':
            if adaptation_status == 'adapted':
                st.success("🏆 **Elite Heat-Adapted**: Outstanding heat fitness! Focus on race tactics and fine-tuning.")
            elif adaptation_status == 'partially_adapted':
                st.info("🏆 **Elite Developing**: Good heat fitness with room for technical improvements.")
            else:
                st.warning("🏆 **Elite Heat-Naive**: Surprising! You have elite speed but need heat adaptation work.")
        elif performance_level in ['very_competitive', 'competitive']:
            if adaptation_status == 'adapted':
                st.success("🏃‍♂️ **Competitive Heat-Adapted**: Excellent heat management for your level!")
            elif adaptation_status == 'partially_adapted':
                st.info("🏃‍♂️ **Competitive Developing**: Good base with solid improvement potential.")
            else:
                st.warning("🏃‍♂️ **Competitive Heat-Naive**: Great fitness foundation, significant heat gains possible!")
        else:
            if adaptation_status == 'adapted':
                st.success("🏃‍♀️ **Recreational Heat-Adapted**: Great job! You handle heat well for your level.")
            elif adaptation_status == 'partially_adapted':
                st.info("🏃‍♀️ **Recreational Developing**: Good progress, lots of potential remaining!")
            else:
                st.warning("🏃‍♀️ **Recreational Heat-Naive**: Huge opportunity! Heat adaptation will transform your summer running.")
    
    return limited_improvement

def calculate_adaptation_days_fixed_v2(improvement_pct, run_data, performance_level=None, adaptation_status=None):
    """
    Calculate days to adaptation plateau based on performance level and adaptation status
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
    
    # Adjust based on adaptation status
    if adaptation_status == "naive":
        base_days += 3  # Heat naive need more time
    elif adaptation_status == "adapted":
        base_days -= 2  # Already mostly there
    
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

def generate_adaptation_advice_fixed_v2(improvement_pct, baseline_hss, plateau_days, run_data, debug_info):
    """Generate advice based on actual performance level and heat adaptation status"""
    
    performance_level = debug_info.get('performance_level', 'unknown')
    adaptation_status = debug_info.get('heat_adaptation_status', 'unknown')
    mile_pr_display = debug_info.get('mile_pr_display', 'Unknown')
    
    # Create category based on BOTH performance level AND adaptation status
    category = f"{performance_level.replace('_', ' ').title()} - {adaptation_status.replace('_', ' ').title()}"
    
    st.subheader("🔥 Fixed Performance-Based Heat Training Plan")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-container"><h3>Runner Profile</h3><p>{category}</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><h3>Improvement Potential</h3><p>{improvement_pct:.1f}%</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><h3>Mile PR</h3><p>{mile_pr_display}</p></div>', unsafe_allow_html=True)
    
    # Universal safety warning
    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
    st.markdown("**⚠️ SAFETY FIRST:**")
    st.markdown("""
    • Heat adaptation must be progressive - start conservatively
    • Monitor for heat illness: dizziness, nausea, confusion, excessive fatigue
    • Proper hydration before, during, and after heat exposure is critical
    • When in doubt, prioritize safety over performance gains
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance and adaptation specific advice
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    
    # Elite runners
    if performance_level == "elite":
        if adaptation_status == "adapted":
            st.markdown(f"**🏆 ELITE HEAT-ADAPTED (PR: {mile_pr_display}):**")
            st.markdown("""
            • **Status:** Peak heat adaptation achieved at elite level
            • **Protocol:** 1-2 heat sessions/week for maintenance
            • **Focus:** Race-specific heat tactics, pacing strategies, cooling techniques
            • **Expected gain:** 1-4% through tactical optimization
            • **Key insight:** You're operating at the ceiling - focus on execution
            """)
        elif adaptation_status == "partially_adapted":
            st.markdown(f"**🏆 ELITE PARTIALLY ADAPTED (PR: {mile_pr_display}):**")
            st.markdown("""
            • **Protocol:** 2-3 targeted heat sessions/week
            • **Focus:** Technical heat management at race pace
            • **Workouts:** Tempo runs and race-pace intervals in 85-90°F
            • **Timeline:** 6-10 days for measurable gains
            • **Expected gain:** 3-6% improvement in heat performance
            """)
        else:  # naive
            st.markdown(f"**🏆 ELITE HEAT-NAIVE (PR: {mile_pr_display}):**")
            st.markdown("""
            • **Opportunity:** Significant gains possible despite elite fitness
            • **Protocol:** Progressive heat exposure, 3-4 sessions/week initially
            • **Phase 1 (Week 1):** Easy runs in moderate heat (80-85°F)
            • **Phase 2 (Week 2-3):** Add structured workouts in heat
            • **Expected gain:** 5-8% - substantial for an elite athlete!
            """)
    
    # Competitive runners
    elif performance_level in ["very_competitive", "competitive"]:
        if adaptation_status == "adapted":
            st.markdown(f"**🏃‍♂️ COMPETITIVE HEAT-ADAPTED (PR: {mile_pr_display}):**")
            st.markdown("""
            • **Status:** Excellent heat fitness for competitive level
            • **Protocol:** 2 heat sessions/week to maintain adaptations
            • **Focus:** Advanced heat racing strategies, nutrition optimization
            • **Expected gain:** 2-8% through refinement and tactics
            • **Strength:** Build on your heat fitness advantage over competitors
            """)
        elif adaptation_status == "partially_adapted":
            st.markdown(f"**🏃‍♂️ COMPETITIVE DEVELOPING (PR: {mile_pr_display}):**")
            st.markdown("""
            • **Week 1-2:** Build base with 45-60min easy runs in warm conditions
            • **Week 2-3:** Add tempo efforts and medium-long runs in heat
            • **Week 3-4:** Integrate race-pace work in target conditions
            • **Expected gain:** 7-12% improvement in heat races
            • **Timeline:** 10-14 days to plateau
            """)
        else:  # naive
            st.markdown(f"**🏃‍♂️ COMPETITIVE HEAT-NAIVE (PR: {mile_pr_display}):**")
            st.markdown("""
            • **Huge potential!** Your competitive fitness + heat adaptation = major gains
            • **Protocol:** Systematic 3-week progression
            • **Week 1:** Base building in moderate heat (75-80°F)
            • **Week 2-3:** Structured workouts, build to race conditions
            • **Expected gain:** 10-15% - this will be transformative!
            """)
    
    # Recreational runners
    else:
        if adaptation_status == "adapted":
            st.markdown(f"**🏃‍♀️ RECREATIONAL HEAT-ADAPTED (PR: {mile_pr_display}):**")
            st.markdown("""
            • **Excellent work!** You've achieved great heat fitness
            • **Protocol:** 2-3 heat runs/week to maintain your advantage
            • **Focus:** Enjoy consistent summer running, explore longer distances
            • **Expected gain:** 3-10% through pacing and fueling improvements
            • **Opportunity:** Consider more challenging heat events/races
            """)
        elif adaptation_status == "partially_adapted":
            st.markdown(f"**🏃‍♀️ RECREATIONAL DEVELOPING (PR: {mile_pr_display}):**")
            st.markdown("""
            • **Good foundation** with excellent potential ahead
            • **Protocol:** Build gradually, 3-4 heat exposures/week
            • **Focus:** Extend time in heat, add light structure
            • **Expected gain:** 12-18% - significant comfort and performance gains
            • **Timeline:** 12-16 days for major improvements
            """)
        else:  # naive
            st.markdown(f"**🏃‍♀️ RECREATIONAL HEAT-NAIVE (PR: {mile_pr_display}):**")
            st.markdown("""
            • **MASSIVE POTENTIAL!** Heat adaptation will revolutionize your running
            • **Week 1:** Start with 20-30min easy runs in 75-80°F
            • **Week 2:** Progress to 45min, add gentle pickups
            • **Week 3-4:** Build to normal training duration in heat
            • **Expected gain:** 18-25% - prepare for a completely different experience!
            • **Key insight:** This represents one of the biggest gains possible in running
            """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'category': category, 
        'performance_level': performance_level,
        'adaptation_status': adaptation_status,
        'advice_level': f"{performance_level}_{adaptation_status}"
    }

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

# USE THE FIXED FUNCTION FOR PACE CONVERSION
def pace_to_seconds(pace_str):
    """Use the fixed pace conversion function"""
    return pace_to_seconds_fixed(pace_str)

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
    st.markdown('<h1 class="main-header">🔥 Heat Adaptation Analysis v3.0 - FIXED</h1>', unsafe_allow_html=True)
    st.markdown("**ML-Enhanced Running Performance Analysis with Proper Performance Level Differentiation**")
    
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
                    pace_sec = pace_to_seconds_fixed(pace_str)
                    mile_pr_sec = pace_to_seconds_fixed(mile_pr_str)
                    
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
                            mile_pr_sec = pace_to_seconds_fixed(mile_pr_str)
                            
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
                                pace_sec = pace_to_seconds_fixed(pace_str_csv)
                                
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
        st.header("🔬 FIXED Heat Adaptation Analysis")
        
        # Analysis configuration
        st.sidebar.subheader("🔧 Analysis Settings")
        outlier_method = st.sidebar.selectbox("Outlier detection method:", ['iqr', 'zscore'])
        outlier_threshold = st.sidebar.slider("Outlier threshold:", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
        
        if st.button("🚀 Run FIXED Analysis", type="primary"):
            with st.spinner("Running FIXED analysis with proper performance level differentiation..."):
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
                    improvement_pct, debug_info = calculate_adaptation_potential_fixed_v2(
                        df, features_df, ml_model, clean_run_data_df
                    )
                    
                    # Apply physiological limits using FIXED function
                    improvement_pct = apply_physiological_limits_fixed_v2(improvement_pct, debug_info=debug_info)
                    
                    # Calculate adaptation days using FIXED function
                    plateau_days = calculate_adaptation_days_fixed_v2(
                        improvement_pct, clean_run_data_df, 
                        debug_info.get('performance_level'), 
                        debug_info.get('heat_adaptation_status')
                    )
                    
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
                    st.subheader("📊 FIXED Analysis Results")
                    
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
                    advice = generate_adaptation_advice_fixed_v2(improvement_pct, baseline_hss, plateau_days, df, debug_info)
                    
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
                            'Heat_Adaptation_Status': debug_info.get('heat_adaptation_status', 'unknown'),
                            'Mile_PR_Display': debug_info.get('mile_pr_display', 'Unknown'),
                            'Improvement_Pct': improvement_pct,
                            'Plateau_Days': plateau_days
                        } for i, run in enumerate(run_data_list)
                    ])
                    
                    csv_data = export_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download FIXED Analysis Results as CSV",
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
    
# REPLACE THE TEST SCENARIOS SECTION WITH THIS UPDATED VERSION
# This fixes the function order issue

# Test scenarios section - FIXED to always be visible
st.sidebar.subheader("🧪 Test the Fixed Logic")

# Always show test scenarios regardless of data state
if st.sidebar.button("Load Test Scenarios"):
    if 'show_test_scenarios' not in st.session_state:
        st.session_state.show_test_scenarios = True
    else:
        st.session_state.show_test_scenarios = not st.session_state.show_test_scenarios

# Show test scenarios interface if activated
if st.session_state.get('show_test_scenarios', False):
    st.header("🧪 Test Scenarios to Verify Fixed Logic")
    
    # Define scenarios inline to avoid function order issues
    test_scenarios = [
        {
            'name': "Elite Heat-Adapted",
            'mile_pr': "4:45",  # 285 seconds - elite
            'runs': [
                {'temp': 85, 'humidity': 70, 'pace': "5:00", 'hr': 165},  # Only 15 sec slower than PR
                {'temp': 90, 'humidity': 75, 'pace': "5:05", 'hr': 170},  # 20 sec slower
                {'temp': 88, 'humidity': 80, 'pace': "5:02", 'hr': 168},  # 17 sec slower
            ],
            'expected_improvement': 3,  # Very low - already adapted elite
            'expected_category': "Elite - Adapted"
        },
        {
            'name': "Elite Heat-Naive", 
            'mile_pr': "4:45",  # 285 seconds - elite
            'runs': [
                {'temp': 85, 'humidity': 70, 'pace': "6:15", 'hr': 180},  # 90 sec slower - very heat naive!
                {'temp': 90, 'humidity': 75, 'pace': "6:30", 'hr': 185},  # 105 sec slower
                {'temp': 88, 'humidity': 80, 'pace': "6:20", 'hr': 183},  # 95 sec slower
            ],
            'expected_improvement': 7,  # Higher - elite but heat naive
            'expected_category': "Elite - Naive"
        },
        {
            'name': "Competitive Heat-Adapted",
            'mile_pr': "5:45",  # 345 seconds - competitive
            'runs': [
                {'temp': 85, 'humidity': 70, 'pace': "6:00", 'hr': 165},  # 15 sec slower - well adapted
                {'temp': 90, 'humidity': 75, 'pace': "6:10", 'hr': 170},  # 25 sec slower
                {'temp': 88, 'humidity': 80, 'pace': "6:05", 'hr': 168},  # 20 sec slower
            ],
            'expected_improvement': 6,  # Low - competitive and adapted
            'expected_category': "Competitive - Adapted"
        },
        {
            'name': "Recreational Heat-Naive",
            'mile_pr': "8:30",  # 510 seconds - recreational  
            'runs': [
                {'temp': 85, 'humidity': 70, 'pace': "11:00", 'hr': 180},  # 150 sec slower - very naive
                {'temp': 90, 'humidity': 75, 'pace': "11:30", 'hr': 185},  # 180 sec slower
                {'temp': 88, 'humidity': 80, 'pace': "11:15", 'hr': 183},  # 165 sec slower
            ],
            'expected_improvement': 23,  # Very high - recreational and heat naive
            'expected_category': "Recreational - Naive"
        }
    ]
    
    selected_scenario = st.selectbox(
        "Choose a test scenario:", 
        [s['name'] for s in test_scenarios]
    )
    
    scenario = next(s for s in test_scenarios if s['name'] == selected_scenario)
    
    # Show scenario details
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Mile PR:** {scenario['mile_pr']}")
        st.write(f"**Expected Category:** {scenario['expected_category']}")
    with col2:
        st.write(f"**Expected Improvement:** {scenario['expected_improvement']}%")
        st.write(f"**Number of Runs:** {len(scenario['runs'])}")
    
    if st.button(f"Load {scenario['name']} Data", type="primary"):
        st.session_state.run_data = []
        mile_pr_sec = pace_to_seconds_fixed(scenario['mile_pr'])
        
        for i, run_data in enumerate(scenario['runs']):
            date_obj = datetime.now() - timedelta(days=len(scenario['runs']) - i)
            pace_sec = pace_to_seconds_fixed(run_data['pace'])
            
            raw_score = heat_score(
                run_data['temp'], run_data['humidity'], 
                pace_sec, run_data['hr'], 190,  # Use a default max HR, 
                3.1, multiplier=1.0
            )
            
            new_run = {
                'date': date_obj,
                'temp': run_data['temp'],
                'humidity': run_data['humidity'],
                'pace_sec': pace_sec,
                'avg_hr': run_data['hr'],
                'max_hr': 190,
                'distance': 3.1,
                'raw_score': raw_score,
                'adjusted_score': None,
                'relative_score': None,
                'mile_pr_sec': mile_pr_sec
            }
            
            st.session_state.run_data.append(new_run)
        
        st.success(f"✅ Loaded {scenario['name']} test data!")
        st.info(f"**Expected:** {scenario['expected_improvement']}% improvement, Category: {scenario['expected_category']}")
        
        # Hide test scenarios after loading and rerun
        st.session_state.show_test_scenarios = False
        st.rerun()
    
    # Footer with information
    st.markdown("---")
    st.markdown("### ℹ️ About This FIXED Analysis")
    with st.expander("🔧 Key Fixes Implemented"):
        st.markdown("""
        **MAJOR FIXES APPLIED:**
        
        **1. Fixed Mile PR Conversion:**
        - OLD BUG: `mile_pr_pace = mile_pr_sec / 60` treated 4:45 mile (285 sec) as "4.75 minutes"
        - FIXED: Proper second-based thresholds: Elite = ≤300sec (5:00), Competitive = ≤360sec (6:00)
        
        **2. Corrected Performance Level Thresholds:**
        - Elite: ≤300 seconds (Sub-5:00 mile)
        - Very Competitive: 300-330 seconds (5:00-5:30)
        - Competitive: 330-360 seconds (5:30-6:00)
        - Good: 360-420 seconds (6:00-7:00)
        - Average: 420-480 seconds (7:00-8:00)
        - Recreational: >480 seconds (8:00+ mile)
        
        **3. Heat Adaptation Status Assessment:**
        - Elite runners: <10% pace slowdown = adapted, >25% = naive
        - Competitive runners: <15% pace slowdown = adapted, >35% = naive  
        - Recreational runners: <20% pace slowdown = adapted, >50% = naive
        
        **4. Proper Improvement Potential:**
        - Elite Heat-Adapted: 1-4% (maintenance focus)
        - Elite Heat-Naive: 5-8% (surprising but possible)
        - Recreational Heat-Adapted: 5-12% (refinement)
        - Recreational Heat-Naive: 18-25% (huge potential)
        
        **5. REMOVED Baseline HSS Override:**
        - The biggest bug was cutting elite improvement in half based on low HSS
        - Now properly differentiates based on performance level + adaptation status
        """)
    
    with st.expander("🧪 Test Results Expected"):
        st.markdown("""
        **Test the fixed logic with these scenarios:**
        
        **Elite Heat-Adapted Runner:**
        - Mile PR: 4:45, Current pace in heat: 5:00-5:05
        - Expected: ~3% improvement, "Elite - Adapted" category
        
        **Elite Heat-Naive Runner:**
        - Mile PR: 4:45, Current pace in heat: 6:15-6:30  
        - Expected: ~7% improvement, "Elite - Naive" category
        
        **Recreational Heat-Adapted Runner:**
        - Mile PR: 8:30, Current pace in heat: 9:00-9:15
        - Expected: ~12% improvement, "Recreational - Adapted" category
        
        **Recreational Heat-Naive Runner:**
        - Mile PR: 8:30, Current pace in heat: 11:00-11:30
        - Expected: ~23% improvement, "Recreational - Naive" category
        
        Use the test scenarios in the sidebar to verify these work correctly!
        """)
    
    with st.expander("How it works"):
        st.markdown("""
        **Heat Strain Score (HSS)** quantifies the physiological stress of running in heat based on:
        - Temperature and humidity conditions (heat index calculation)
        - Your running pace vs. personal record (performance stress)
        - Heart rate response (cardiovascular stress)
        - Distance covered (duration factor)
        
        **Fixed Machine Learning Models** now properly analyze:
        - Performance level assessment (elite vs recreational capability)
        - Heat adaptation status (adapted vs naive response patterns)
        - Realistic improvement potential based on both factors
        - Confidence intervals for predictions
        
        **Key Features:**
        - Proper performance level differentiation
        - Heat adaptation status assessment
        - Outlier detection and filtering
        - Multiple modeling approaches (Random Forest, Gradient Boosting)
        - Physiologically-constrained improvement estimates
        - Performance-level-based training recommendations
        """)

if __name__ == "__main__":
    main()
