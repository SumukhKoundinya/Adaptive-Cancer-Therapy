ACTION_SPACE = [
   "PD1_LOW_DOSE",
   "PD2_HIGH_DOSE",
   "COMBINATION_THERAPY",
   "NO_TREATMENT", 
]

def encode_action(action_name):
    if action_name == "PD1_LOW_DOSE":
        return {"efficacy": 0.4, "pressure": 0.3}
    
    if action_name == "PD1_HIGH_DOSE":
        return {"efficacy": 0.7, "pressure": 0.6}
    
    if action_name == "COMBINATION_THERAPY":
        return {"efficacy": 0.9, "pressure": 0.9}
    
    if action_name == "NO_TREATMENT":
        return {"efficacy": 0.0, "pressure": 0.0}
