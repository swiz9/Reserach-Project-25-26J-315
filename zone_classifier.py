"""
Zone Classification for Arrhythmia Detection
Provides clinical risk zone classification based on predicted arrhythmia types.
"""

def default_zone_policy(predicted_label, confidence):
    """
    Default policy for classifying arrhythmia predictions into clinical zones.
    
    Args:
        predicted_label (str): Predicted arrhythmia class ('E', 'F', 'N', 'V')
        confidence (float): Model confidence score (0-1)
    
    Returns:
        tuple: (zone, explanation)
    """
    
    # Define zone classifications based on arrhythmia type and confidence
    if predicted_label == 'N':  # Normal
        if confidence > 0.8:
            zone = "Green Zone"
            explanation = "Normal heart rhythm with high confidence"
        else:
            zone = "Yellow Zone" 
            explanation = "Normal rhythm but low confidence - recommend review"
    
    elif predicted_label == 'V':  # Ventricular premature contraction
        if confidence > 0.7:
            zone = "Red Zone"
            explanation = "Ventricular arrhythmia detected - immediate attention required"
        else:
            zone = "Yellow Zone"
            explanation = "Possible ventricular activity - further monitoring needed"
    
    elif predicted_label == 'F':  # Fusion beats
        zone = "Yellow Zone"
        explanation = "Fusion beats detected - requires clinical evaluation"
    
    elif predicted_label == 'E':  # Ventricular escape
        zone = "Red Zone"
        explanation = "Ventricular escape rhythm - clinical attention required"
    
    else:
        zone = "Yellow Zone"
        explanation = "Unknown rhythm pattern - requires further investigation"
    
    return zone, explanation

def get_zone_color(zone):
    """Get color code for zone display"""
    colors = {
        "Green Zone": "Green",
        "Yellow Zone": "Yellow", 
        "Red Zone": "Red"
    }
    return colors.get(zone, "Unknown")
