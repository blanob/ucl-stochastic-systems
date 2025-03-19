import numpy as np

# Set up the explanatory component
def checkpoint(freq, n):
    checkpoints = np.arange(freq**-1, n, freq**-1)
    return checkpoints

# Set up the stochastic component
def expFailure(rFailure):
    tFailure = np.random.exponential(rFailure**-1)
    return tFailure

def expRepair(rRepair):
    tRepair = np.random.exponential(rRepair**-1)
    return tRepair

# A Function to implement a single step in a renewal process
def single_renewal(freq, n, rFailure, rRepair):
    
    checkpoints = checkpoint(freq, n)
    
    tFailure = expFailure(rFailure)
    tRepair = expRepair(rRepair)
    
    online = tFailure
    offline = 0
    
    for i in range(0, len(checkpoints)):
        if tFailure < checkpoints[i]:
            offline += checkpoints[i]-tFailure+repair
            break
        elif tFailure > checkpoints[i]:
            offline += 0
            continue
        
    return online, offline

# A Function to implement a simple renewal process
def renewal(freq, n, rFailure, rRepair):
    
    checkpoints = checkpoint(freq, n)
    
    online = []
    offline = []
    
    runtime = 0
    repair_old = 0
    
    while runtime < checkpoints[-1]:
        
        tFailure = expFailure(rFailure)
        tRepair = expRepair(rRepair)
        online.append(tFailure)
        
        for i in range(0, len(checkpoints)):
            if tFailure < checkpoints[i] - repair_old:
                offline.append(checkpoints[i] - repair_old - tFailure + repair)
                break
            elif tFailure > checkpoints[i] - repair_old:
                offline.append(0)
                continue
                
        repair_old = tRepair
        runtime = sum(online + offline)
        
    return online + offline
