import numpy as np

class LinfinityRegression:

    def __init__(self, iterations, learning_rate, regularization_strength):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength

    @staticmethod
    def soft_thresholding_operator(x, l):
        """
        This method is used to update the weights when performing Gradient Descent.
        Whenever the loss function is just the least square loss function, we can minimize by taking the derivative.
        However, we cannot minimise the Lasso Loss function in the same weight because the function is not differentiable
        at w = 0 (where w is any of the weight component)


        :param x:
        :param l:
        :return:
        """
        maxw = max(x)
        for i in range(0, len(x)):
            if np.abs(x[i]) > np.abs(maxw):
                x[i] = l*np.sign(x[i])
            elif np.abs(x[i]) < np.abs(maxw) :
                x[i]  = l*np.sign(maxw)
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
        weights = np.random.rand(training_records.shape[1])
        weights_table = [weights]
        predicted_outputs = []
        for i in range(self.iterations):
            predicted_output = np.dot(training_records, weights)
            predicted_outputs.append(predicted_output)
            mse_cost, error = LinfinityRegression.mse_cost_function(predicted_output, output)
            mse_costs.append(mse_cost)
            slope = training_records.T.dot(error)/(len(output))
            weights = LinfinityRegression.soft_thresholding_operator(weights - self.learning_rate*slope,
                                                                 self.regularization_strength)
            weights_table.append(weights.copy())
        return weights_table, mse_costs, predicted_outputs