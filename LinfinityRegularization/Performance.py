import numpy as np
from sklearn.datasets.samples_generator import make_regression
from sklearn import linear_model
from LinfinityRegularizer import LinfinityRegression
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sea
import matplotlib.patches as mpatches
from sklearn.datasets.samples_generator import make_regression
from mpl_toolkits.mplot3d import Axes3D


# Define synthetic data-set constants. Change this to experiment with different data sets
NUM_OF_SAMPLES = 2000
NUM_OF_FEATURES = 2
NOISE = 20

# Define the number of iterations and learning rate for Linear regression.
NUM_OF_ITERATIONS =1000
LEARNING_RATE = 0.1
LINFINITY_LEARNING_RATE = 1.0


def calculate_weights(training_records, output):
    mse_costs = []
    weights = np.random.rand(training_records.shape[1])
    weights_table = [weights]
    predicted_outputs = []
    itr = 0
    prevErr = 0
    for i in range(NUM_OF_ITERATIONS):
        predicted_output = np.dot(training_records, weights)
        predicted_outputs.append(predicted_output)
        mse_cost, error = LinfinityRegression.mse_cost_function(predicted_output, output)
        mse_costs.append(mse_cost)
        slope = training_records.T.dot(error) / (len(output))
        weights = LinfinityRegression.soft_thresholding_operator(weights - LEARNING_RATE * slope,
                                                                 LINFINITY_LEARNING_RATE)
        weights_table.append(weights.copy())
        if (abs(prevErr-mse_cost)<0.001):
                itr = i
                return itr,mse_costs
        prevErr = mse_cost
    return itr,mse_costs



def start_Linfinity_regression(training_records, output):
    itr,mse_costs = calculate_weights(training_records, output)
    return itr,mse_costs


learningSpeeds = []

for i in range(0,100):
    # generate sample data-set using the following function.
    training_rec, out = make_regression(n_samples=NUM_OF_SAMPLES,
                                        n_features=NUM_OF_FEATURES,
                                        n_informative=1,
                                        noise=NOISE)

    # Add a columns of 1s as bias(intercept) in the training records
    training_rec = np.c_[np.ones(training_rec.shape[0]), training_rec]
    itr,mse_costs = start_Linfinity_regression(training_records=training_rec, output=out)
    print "Loop ",i,": Weights converged at iteration #",itr
    print "Initial MSE cost: ",mse_costs[0]," Final MSE cost: ",mse_costs[len(mse_costs)-1]
    learningSpeeds.append(itr)

avgSpeed = sum(learningSpeeds)/len(learningSpeeds)
print "Average learning speed for Linf Regression is: ", avgSpeed


#PLOTS
x = np.arange(0,100,1)
y = learningSpeeds
plt.bar(x, y, 0.2, color='b', label='Learning Speed - Linf Regression')
plt.errorbar(x, y, xerr=0, yerr=0)


plt.xlabel('Iterations on Different Datasets')
plt.title('Linf Regression - Learning Speed\nAverage speed is {:.2f} iterations for Alpha = {:.5f}'.format(avgSpeed, LEARNING_RATE))
plt.ylabel('Convergence Iteration')

plt.axhline(avgSpeed, color='r', linestyle='--')
plt.show()

