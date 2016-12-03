import numpy as np
from sklearn.datasets.samples_generator import make_regression
from Lp import Lp

# Define synthetic data-set constants. Change this to experiment with different data sets
NUM_OF_SAMPLES = 1000
NUM_OF_FEATURES = 2
NOISE = 10

# Define the number of iterations and learning rate for Linear regression.
NUM_OF_ITERATIONS = 2000
LEARNING_RATE = 0.01
LP_REGULARIZATION_STRENGTH = 1.0

# generate sample data-set using the following function.
training_rec, out = make_regression(n_samples=NUM_OF_SAMPLES,
                                    n_features=NUM_OF_FEATURES,
                                    n_informative=1,
                                    noise=NOISE)



# Add a columns of 1s as bias(intercept) in the training records
training_rec = np.c_[np.ones(training_rec.shape[0]), training_rec]


def start_lp_regression(training_records, output):
    """
    In this method, we compare the weights calculated using our gradient descent approach with the sklearn's output.

    `Our method`
    >>> regressor = Lp(iterations=NUM_OF_ITERATIONS, learning_rate=LEARNING_RATE, regularization_strength=LP_REGULARIZATION_STRENGTH)
    >>> weights_table, mse_costs, predicted_outputs = regressor.calculate_weights(training_records, output)

    As you see above there are 3 tables returned from our approach.

    1. weights_table - This is where we store the history of the weights from iteration 0 to the last iteration.
       To access the set of weights in the last iteration simply use `weights_table[-1]`

    2. mse_costs - Table which stores the mean square error for each iteration.

    3. predicted_outputs - This is the predicted output using our machine(i.e weights)

    The following code fragment shows how to invoke sklearn's Lasso regression.
    `sklearn's method`
    >>> clf = linear_model.Lasso(fit_intercept=False)
    >>> clf.fit(training_records, output)

    Lastly, we just print the weights and it is left to the user to visually compare them.

    :parameter training_records - N X P matrix of training samples.
    :parameter output - N X 1 vector of output.

    :return:
    """
    regressor = Lp(iterations=NUM_OF_ITERATIONS, learning_rate=LEARNING_RATE, regularization_strength=LP_REGULARIZATION_STRENGTH)
    weights_table, mse_costs, predicted_outputs = regressor.calculate_weights(training_records, output)

    print "Starting gradient descent with {0} iterations, learning rate of {1} and a regularization " \
          "strength of {2}".format(NUM_OF_ITERATIONS, LEARNING_RATE, LP_REGULARIZATION_STRENGTH)

    print "Running..."

    final_weights = [weights_table[-1][i] for i in range(0, NUM_OF_FEATURES+1)]
    print "After %s iterations of Gradient Descent (our implementation), the final weights are : %s" % (NUM_OF_ITERATIONS, final_weights)
    print mse_costs[0], mse_costs[-1]
    print weights_table[0], weights_table[-1]

start_lp_regression(training_records=training_rec, output=out)









