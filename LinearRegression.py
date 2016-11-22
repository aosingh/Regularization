import numpy as np


class LinearRegression:
    """
    This class is responsible for performing Linear Regression using the gradient descent approach.
    """

    def __init__(self, iterations, learning_rate, regularizer=None):
        '''

        :param iterations: The number of iterations before convergence

        :param learning_rate: How much do we want to move(in each iteration) in the direction of negative gradient
                              of the cost function. A value of 0 means not at all.

        :param regularizer: 1 for L1,
                            2 for L2,
                            'inf' for Linf
                            0<p<1 for Lp
                            default is `None`
        :return:
        '''
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularizer = regularizer

    @staticmethod
    def mse_cost_function(predicted_output, actual_output):
        """
        This method calculates the error and the MSE cost function given a predicted_value and the actual_value


        :param predicted_output:
        :param actual_output:

        :return: Mean Square Error, Error.
        """
        error = predicted_output - actual_output
        mse_cost = np.sum(error ** 2) /(2 * len(actual_output))
        return mse_cost, error

    def calculate_weights(self, training_record, output):
        """
        This method is responsible for calculating weights or coefficients using the gradient descent approach.

        Please read this link to understand gradient descent in detail : https://en.wikipedia.org/wiki/Gradient_descent

        The basic idea is calculate a cost function and then move in the direction of negative gradient at each step.

        Finally, after certain number of iterations we converge and achieve the minimum value of the cost function.
        In our case the cost function that we are trying to minimize is the MEAN SQUARE ERROR.


        :param training_record:
        :param output:
        :return:

        1. weights_table - This is where we store the history of the weights from iteration 0 to the last iteration.
           To access the set of weights in the last iteration simply use `weights_table[-1]`

        2. mse_costs - Table which stores the mean square error for each iteration.

        3. predicted_outputs - This is the predicted output using our machine(i.e weights)

        """
        mse_costs = []
        weights = np.random.rand(training_record.shape[1])
        weights_table = [weights]
        predicted_outputs = []
        for i in range(self.iterations):
            predicted_output = np.dot(training_record, weights)
            predicted_outputs.append(predicted_output)
            mse_cost, error = LinearRegression.mse_cost_function(predicted_output, output)
            mse_costs.append(mse_cost)
            slope = training_record.T.dot(error)/(len(output))
            weights -= (self.learning_rate * slope)
            weights_table.append(weights)
        return weights_table, mse_costs, predicted_outputs








