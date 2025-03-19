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
#  SIMULATION
#
#################################################################################################################################################

freq_val = np.linspace()


#################################################################################################################################################
#
#  POLYNOMIAL REGRESSION of simulated data
#
#################################################################################################################################################

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

degrees = [1, 2, 3]

X = np.sort(freq_val)
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