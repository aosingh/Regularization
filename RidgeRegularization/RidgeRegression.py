import numpy as np

class RidgeRegression:

    def __init__(self, learning_rate, ridge_learning_rate, iterations):
        self.learning_rate = learning_rate
        self.ridge_learning_rate = ridge_learning_rate
        self.iterations = iterations



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
        weights = np.random.rand(training_records.shape[1])
        weights_table = [weights]
        predicted_outputs = []
        for i in range(self.iterations):
            predicted_output = np.dot(training_records, weights)
            predicted_outputs.append(predicted_output)
            mse_cost, error = RidgeRegression.mse_cost_function(predicted_output, output)
            mse_costs.append(mse_cost)
            slope = training_records.T.dot(error)/(len(output))
            weights -= (self.learning_rate * (slope + (self.ridge_learning_rate/len(output))*weights))
            weights_table.append(weights.copy())
        return weights_table, mse_costs, predicted_outputs



