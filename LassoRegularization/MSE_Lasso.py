import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.datasets.samples_generator import make_regression
from LassoRegressionTester import start_lasso_regression
from mpl_toolkits.mplot3d import Axes3D



NUM_OF_SAMPLES = 200
NUM_OF_FEATURES = 2
NOISE = 10

# Define the number of iterations and learning rate for Linear regression.
NUM_OF_ITERATIONS = 1000
LEARNING_RATE = 0.01
LASSO_REGULARIZATION_STRENGTH = 1.0

# generate sample data-set using the following function.
training_rec, out = make_regression(n_samples=NUM_OF_SAMPLES,
                                    n_features=NUM_OF_FEATURES,
                                    n_informative=1,
                                    noise=NOISE)

# Add a columns of 1s as bias(intercept) in the training records
training_rec = np.c_[np.ones(training_rec.shape[0]), training_rec]

weights_table, MSEcost = start_lasso_regression(training_records=training_rec, output=out)
itr = []
w1 = []
w2 = []

for i in range(0,len(weights_table)-1):
    itr.append(i)
    #if not (i==0):
    print "w1: ", weights_table[i][1]
    w1.append(weights_table[i][1])
    w2.append(weights_table[i][2])
    print "w2: ", weights_table[i][2]
    print MSEcost[i]

#plot of error through each gradient descent iteration
x = itr
y = MSEcost
plt.errorbar(x, y, xerr=0, yerr=0)
blue_patch = mpatches.Patch(color='blue', label="MSE Error for L1 Regularizer", )
plt.legend(handles=[blue_patch], )
plt.xlabel('Iterations')
plt.ylabel('MSE cost')
plt.show()


#plot error versus w1 and w2
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
ax = fig.gca(projection='3d')
x = w1
y = w2
z = MSEcost
ax.plot(x, y, z, label='MSE Error curve for L1 Regularizer - In coefficient Space')
ax.legend()
plt.xlabel('Coefficient 1')
plt.ylabel('Coefficient 2')
plt.show()

