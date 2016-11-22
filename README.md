# Regularization
In this project we learn various regularizers with using Linear regression as a tool.

### Clone this Repo
Clone this repository into whatever directory you'd like to work on it from:

```bash
git clone https://github.com/aosingh/Regularization.git
```

### Install the following on your local system
*   [Python 2.7](https://www.python.org/download/releases/2.7/)
*   [pandas](http://pandas.pydata.org/)
    *   `pip install pandas`
*   [scikit-learn](http://scikit-learn.org/stable/)
    *   `pip install -U scikit-learn`
    
### Files 

* `LinearRegression.py`
This class is responsible for performing [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) using the [gradient descent approach](https://en.wikipedia.org/wiki/Gradient_descent).

    * The basic idea is calculate a cost function and then move in the direction of negative gradient at each step.
      Finally, after certain number of iterations we converge and achieve the minimum value of the cost function.
      In our case the cost function that we are trying to minimize is the MEAN SQUARE ERROR.


* `LinearRegressionTester.py`
In this file we invoke our Linear Regression method defined in the above class and compare the output with sklearn's version.
     
