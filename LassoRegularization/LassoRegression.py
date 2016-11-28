import numpy as np

class LassoRegression:

    def __init__(self, iterations, learning_rate, regularization_strength):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength

    @staticmethod
    def soft_thresholding_operator(x, l):
        for i in range(0, len(x)):
            if x[i] > l:
                x[i] -= l
            elif x[i] < -l:
                x[i] += l
            else: x[i] = 0
        return x



    @staticmethod
    def mse_cost_function(predicted_output, actual_output):
        """
        This method calculates the error and the MSE cost function given a predicted_value and the actual_value


        :param predicted_output:
        :param actual_output:

        :return: Mean Square Error, Error.
        """
        error = predicted_output - actual_output
        mse_cost = np.sum(error ** 2) /(2*len(actual_output))
        return mse_cost, error

    def calculate_weights(self, training_records, output):
        mse_costs = []
        weights = np.zeros(training_records.shape[1])
        weights_table = [weights]
        predicted_outputs = []
        for i in range(self.iterations):
            predicted_output = np.dot(training_records, weights)
            predicted_outputs.append(predicted_output)
            mse_cost, error = LassoRegression.mse_cost_function(predicted_output, output)
            mse_costs.append(mse_cost)
            slope = training_records.T.dot(error)/(len(output))
            weights = LassoRegression.soft_thresholding_operator(weights - self.learning_rate*slope,
                                                                 self.learning_rate*self.regularization_strength)
            weights_table.append(weights)
        return weights_table, mse_costs, predicted_outputs