import numpy as np

# Set up the explanatory component
def checkpoint(freq, n):
    checkpoints = np.arange(freq**-1, n, freq**-1)
    return checkpoints

# Set up the stochastic component
def expFailure(rFailure):
    tFailure = np.random.exponential(rFailure**-1)
    return tFailure

def expInspection(rInspection):
    tInspection = np.random.exponential(rInspection**-1)
    return tInspection

def expRepair(rRepair):
    tRepair = np.random.exponential(rRepair**-1)
    return tRepair

# Function for a renewal process
def inspectionProcess(freq, n, rFailure, rInspection, rRepair):
    
    checkpoints = checkpoint(freq, n)
    
    # Set up the response component
    uptime = []
    downtime = []
    
    # Initialise key logging timepoints
    tRenewal = 0
    tResidual = 0 # spillover time at inspection point
    
    while tRenewal < checkpoints[-1]:
        
        tFailure = expFailure(rFailure)
        
        for i in range(0, len(checkpoints)):
            
            if tRenewal + tFailure > checkpoints[i]:
                # No failure occurs **before the checkpoint**, Inspection conducted.
                tInspection = expInspection(rInspection)
                
                if tRenewal + tFailure > checkpoints[i] + tInspection:
                    # No failure occurs **during the inspection**, Loop continues
                    uptime.append(freq**-1 - tResidual)
                    downtime.append(tInspection)
                    tResidual = tInspection
                    continue
                elif tRenewal + tFailure < checkpoints[i] + tInspection:
                    # Failure occurs **during the inspection**, Repair performed, Loop breaks.
                    tRepair = expRepair(rRepair)
                    tLatent = 0 # No latent time
                    uptime.append(freq**-1 - tResidual - tLatent)
                    downtime.append(tLatent + tInspection + tRepair)
                    tResidual = tInspection + tRepair
                    tRenewal = tRenewal + tFailure + tLatent + tInspection + tRepair
                    break
        
            elif tRenewal + tFailure < checkpoints[i]:
                # Failure occurs, Inspection conducted, Repair performed, Loop breaks.
                tInspection = expInspection(rInspection)
                tRepair = expRepair(rRepair)
                tLatent = checkpoints[i] - (tRenewal + tFailure)
                uptime.append(freq**-1 - tResidual - tLatent)
                downtime.append(tLatent + tInspection + tRepair)
                tResidual = tInspection + tRepair
                tRenewal = tRenewal + tFailure + tLatent + tInspection + tRepair
                break
                
    return uptime, downtime
    