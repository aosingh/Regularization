import numpy as np
from sklearn.datasets.samples_generator import make_regression
from sklearn import linear_model
from RidgeRegressionTester import start_ridge_regression
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sea
import matplotlib.patches as mpatches
from sklearn.datasets.samples_generator import make_regression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D
from pprint import pprint
import random



# Define synthetic data-set constants. Change this to experiment with different data sets
NUM_OF_SAMPLES = 2000
NUM_OF_FEATURES = 2
NOISE = 10

# Define the number of iterations and learning rate for Linear regression.
NUM_OF_ITERATIONS =2000
LEARNING_RATE = 0.01

# generate sample data-set using the following function.
x, y = make_regression(n_samples=NUM_OF_SAMPLES,
                                    n_features=NUM_OF_FEATURES,
                                    n_informative=1,
                                    noise=NOISE)

# Add a columns of 1s as bias(intercept) in the training records
x = np.c_[np.ones(x.shape[0]), x]
print np.shape(x)
print np.shape(y)
weights = []

for i in range(0,100):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=int(random.uniform(1, 76)))
    print np.shape(x_train)
    print np.shape(y_train)
    weight_table, MSEcost = start_ridge_regression(x_train,y_train)
    weights.append(weight_table[-1]);
pprint(weights[1])

weights1 = [rows[0] for rows in weights]
weights2 = [rows[1] for rows in weights]
weights3 = [rows[2] for rows in weights]

variance1 = np.std(weights1)
variance2 = np.std(weights2)
variance3 = np.std(weights3)

mean1 = np.mean(weights1)
mean2 = np.mean(weights2)
mean3 = np.mean(weights3)
print "Stability of the L2 Regularizer 100 iterations are = {:2e}(+/-{:.2e}), {:2e}(+/-{:2e}), {:2e}(+/-{:2e}))".format(mean1, variance1, mean2, variance2, mean3, variance3)


