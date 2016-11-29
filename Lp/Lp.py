import numpy as np
import math

class Lp:
    def __init__(self, iterations, learning_rate, regularization_strength):
        self.p = 0.5
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength


    @staticmethod
    def thresholding_filtering_function(x, l):
        print x
        print l
        m = (float(2)/3)*math.acos((l/8)*math.pow(np.abs(x)/3, float(-3/2)))
        n = np.cos((float(2)*np.pi/3) - m)
        return float(2/3)*x*(1 + n)



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
        for i in range(0, len(x)):
            threshold = (float(3)/4)*math.pow(2, float(1)/3)*math.pow(l, float(2)/3)
            if np.abs(x[i]) > threshold:
                x[i] = Lp.thresholding_filtering_function(x[i], l)
            else:
                x[i] = 0
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
            mse_cost, error = Lp.mse_cost_function(predicted_output, output)
            mse_costs.append(mse_cost)
            slope = training_records.T.dot(error)/(len(output))
            weights = Lp.soft_thresholding_operator(weights - self.learning_rate*slope,
                                                                 self.learning_rate*self.regularization_strength)
            weights_table.append(weights)
        return weights_table, mse_costs, predicted_outputs