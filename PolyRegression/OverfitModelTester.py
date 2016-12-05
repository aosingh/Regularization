from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets.samples_generator import make_regression
from RidgeRegularization import RidgeRegression
from LinearRegression import LinearRegression
from RidgeRegularization import RidgeRegression
from LassoRegularization import LassoRegression
from Lp import Lp
import matplotlib.pyplot as plt
import random

from sklearn import linear_model

import numpy as np

# Define synthetic data-set constants. Change this to experiment with different data sets
NUM_OF_SAMPLES = 100
NUM_OF_FEATURES = 1
NOISE = 10


# Define the number of iterations and learning rate for Linear regression.
NUM_OF_ITERATIONS = 100
LEARNING_RATE = 0.00001

LINEAR_NUM_OF_ITERATIONS = 100
LINEAR_LEARNING_RATE = 0.00001

LASSO_NUM_OF_ITERATIONS = 100
LASSO_LEARNING_RATE = 0.00001



def f(x):
    return x*x

x = np.reshape(np.linspace(0, 20, 1000), (-1, 1))
x  = x.astype(dtype='float64')
y = np.reshape(f(x), (1000, ))




# generate sample data-set using the following function.
training_rec, out = make_regression(n_samples=NUM_OF_SAMPLES,
                                    n_features=NUM_OF_FEATURES,
                                    n_informative=1,
                                    noise=NOISE)
#x = training_rec
#x_plot = np.c_[np.ones(x.shape[0]), x]

poly = PolynomialFeatures(2, include_bias=True, interaction_only=False);
new_features = np.array(poly.fit_transform(x, y=y))




def start_overfit_tester():
    regressor = LinearRegression.LinearRegression(iterations=NUM_OF_ITERATIONS, learning_rate=LINEAR_LEARNING_RATE)

    ridge_regression = RidgeRegression.RidgeRegression(iterations=NUM_OF_ITERATIONS, learning_rate=LEARNING_RATE, ridge_learning_rate=1)

    lasso_regression = LassoRegression.LassoRegression(iterations=LASSO_NUM_OF_ITERATIONS, learning_rate=LASSO_LEARNING_RATE, regularization_strength=1)

    lp = Lp.Lp(iterations=100, learning_rate=0.00001, regularization_strength=1)

    lweights_table, lmse_costs, lpredicted_outputs = lasso_regression.calculate_weights(new_features, y)
    rweights_table, rmse_costs, rpredicted_outputs = ridge_regression.calculate_weights(new_features, y)
    weights_table, mse_costs, predicted_outputs = regressor.calculate_weights(new_features, y)
    lpweights_table, lpmse_costs, lppredicted_outputs = lp.calculate_weights(new_features, y)






    #print weights_table[-1]
    #print predicted_outputs[-1]
    #print out

    clf = linear_model.Lasso(fit_intercept=False)
    clf.fit(new_features, y)
    sklearn_outputs = clf.predict(new_features)

    #print x
    #print y

    plt.scatter(x, y, color='cornflowerblue', s=30, marker='o',  label='Training data')
    #print predicted_outputs[-1]
    plt.plot(x, predicted_outputs[-1], color='red', linewidth=2 , label='Polynomial Regression. MSE is {0}'.format(mse_costs[-1]))

    plt.plot(x, rpredicted_outputs[-1], color='black', linewidth=2 , label='L2 Regularizer. MSE is {0}'.format(rmse_costs[-1]))
    plt.plot(x, lpredicted_outputs[-1], color='pink', linewidth=2, label='L1 Regualrizer. MSE is {0}'.format(lmse_costs[-1]))
    plt.plot(x, lppredicted_outputs[-1], color='orange', linewidth=2, label='Lp(p =0.5) Regularizer is {0}'.format(lpmse_costs[-1]))
    plt.title("Curve Fitting for Learning rate = {0} and iterations = {1}".format(LEARNING_RATE, NUM_OF_ITERATIONS))
    plt.xlabel("X")
    plt.ylabel("x*x")
    plt.legend()
    plt.show()

    #print clf.coef_
    #print sklearn_outputs

start_overfit_tester()