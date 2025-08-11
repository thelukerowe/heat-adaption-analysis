"""
Heat Adaptation Advice Generator

Provides personalized heat adaptation training recommendations based on analysis results.
"""

from typing import List, Dict


def generate_adaptation_advice(improvement_pct: float, baseline_hss: float, 
                             plateau_days: int, run_data: List[Dict]) -> Dict[str, str]:
    """Generate personalized heat adaptation training advice"""
    print("\n" + "="*60)
    print("🔥 HEAT ADAPTATION TRAINING RECOMMENDATIONS 🔥")
    print("="*60)
    
    # Determine adaptation category
    if improvement_pct >= 18:  # Lowered from 20
        category = "Heat-Naive (High Adaptation Potential)"
    elif improvement_pct >= 12:  # Lowered from 15
        category = "Moderately Heat-Adapted"
    elif improvement_pct >= 8:   # Lowered from 10
        category = "Somewhat Heat-Adapted"
    else:
        category = "Well Heat-Adapted"
    
    print(f"📊 Current Heat Adaptation Status: {category}")
    print(f"📈 Expected Improvement Potential: {improvement_pct:.1f}%")
    print(f"⏱️  Estimated Time to Plateau: {plateau_days} days")
    
    # Universal recommendations
    print(f"\n⚠️  UNIVERSAL PRECAUTIONS:")
    print(f"   • Start conservatively and progress gradually")
    print(f"   • Stop if you experience dizziness, nausea, or excessive fatigue")
    print(f"   • Increase fluid intake starting 2-3 days before heat exposure")
    print(f"   • Consider electrolyte replacement during/after heat sessions")
    print(f"   • Allow 48-72 hours between intense heat adaptation sessions")
    
    print(f"\n🌡️  HEAT ADAPTATION TIMELINE:")
    print(f"   • Days 1-3: Initial physiological responses begin")
    print(f"   • Days 4-7: Plasma volume expansion (easier sweating)")
    print(f"   • Days 8-12: Improved sweat rate and cooling efficiency")
    
    print(f"\n💡 PROGRESS MONITORING:")
    print(f"   • Track RPE (Rate of Perceived Exertion) during heat sessions")
    print(f"   • Monitor heart rate response to same-intensity heat exercise")
    print(f"   • Watch for improved comfort in hot conditions")
    print(f"   • Re-run this analysis after {plateau_days} days to verify improvement")
    
    return {
        'category': category
    }
