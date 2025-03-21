#################################################################################################################################################
#
#  TEST RUN `inspectionFunction()` with empirical parameters
#
#  See van der Weide, J. A. M., and Mahesh D. Pandey. 2015. “A Stochastic Alternating Renewal Process Model for Unavailability Analysis of
#  Standby Safety Equipment.” Reliability Engineering and System Safety. https://doi.org/10.1016/j.ress.2015.03.005.
#
#  In the paper, the 40-month time average unavailability is calculated as 0.1238.
#
#################################################################################################################################################

%run processFunctions.py

freq = (4*30)**-1
n = 40*30
rFailure = (17.72*30)**-1
rInspection = 3
rRepair = (0.5*30)**-1

uptime, downtime = inspectionProcess(freq, n, rFailure, rInspection, rRepair)

na = sum(uptime)/sum(uptime+downtime)

#################################################################################################################################################
#
#  SIMULATION & OPTIMISATION
#
#################################################################################################################################################

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
                
    na = sum(downtime)/sum(uptime+downtime)
    return na

grp_n = 10
intervals = np.linspace(1/3, 365, 365)
results = {i: [inspectionProcess(i**-1, n=10*365, rFailure=(17.72*30)**-1, rInspection=3, rRepair=7**-1) for _ in range(grp_n)] for i in intervals}

i_val = []
na_val = []
for i, values in results.items():
    i_val.extend([i] * grp_n)
    na_val.extend(values)
    
mean_na = []
for i in range(0, len(intervals)):
    mean_na.append(np.mean(na_val[i*grp_n: (i+1)*grp_n]))

#################################################################################################################################################
#
#  POLYNOMIAL REGRESSION
#
#################################################################################################################################################

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

degrees = [1, 2, 4]

X = np.sort(i_val)
y = na_val

for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline(
        [
            ("polynomial_features", polynomial_features),
            ("linear_regression", linear_regression),
        ]
    )
    pipeline.fit(X[:, np.newaxis], y)
    
    # Evaluate the models using crossvalidation
    scores = cross_val_score(
        pipeline, X[:, np.newaxis], y, scoring="neg_mean_squared_error", cv=10
    )
    
    X_test = np.linspace(0, 240, 1000)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.scatter(X, y, edgecolor="b", s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    plt.title(
        "Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
            degrees[i], -scores.mean(), scores.std()
        )
    )

plt.show()
    
#################################################################################################################################################
#
#  CONTOUR VISUALISATION
#
#################################################################################################################################################

from matplotlib import cm

freq_val = np.linspace(1, 60, 100)
mu_val = np.linspace(0.3, 1, 8)

xs, ys = np.meshgrid(freq_val, mu_val)

zs = np.array([inspectionProcess(x**-1, 10*365, (17.72*30)**-1, y**-1, 7**-1) for x, y in zip(xs.flatten(), ys.flatten())])
zs = zs.reshape(xs.shape)

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.3))

# =============
# First subplot
# =============
# set up the Axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface
surf = ax.plot_surface(xs, ys, zs, vmin=zs.min() * 2, cmap='coolwarm',
                       linewidth=0, antialiased=False)
ax.set_zlim(0, 1.0)
fig.colorbar(surf, shrink=0.5, aspect=10)
ax.set_title("(a) 3D Surface Plot")
ax.set_xlabel('$f^{-1}$')
ax.set_ylabel('$\mu^{-1}$')

# ==============
# Second subplot
# ==============
# set up the Axes for the second plot
ax = fig.add_subplot(1, 2, 2)

# plot a contour
levels = np.linspace(zs.min(), zs.max(), 10)
contour = ax.contourf(xs, ys, zs, levels=levels, cmap='coolwarm')
cbar = fig.colorbar(contour, shrink=0.5, aspect=10)
ax.set_title("(b) Contour Map")
ax.set_xlabel('$f^{-1}$')
ax.set_ylabel('$\mu^{-1}$')
cbar.set_label('unavailability')

plt.show()