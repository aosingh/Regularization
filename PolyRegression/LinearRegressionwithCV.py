import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

np.random.seed(0)

samples = 30
degrees = [1, 5, 15]

func = lambda X: np.sin(2.5 * np.pi * X * X)
training_set = np.sort(np.random.rand(samples))
y = func(training_set) + np.random.randn(samples) * 0.1

plt.figure(figsize=(20, 6))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=False)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

    pipeline.fit(training_set[:, np.newaxis], y)

    scores = cross_val_score(pipeline, training_set[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=20)

    X_test = np.linspace(0, 1, 100)
    plt.plot(X_test, pipeline.predict(X_test[:, np.newaxis]), label="Model")
    plt.plot(X_test, func(X_test), label="True function")
    plt.scatter(training_set, y, label="Samples")
    plt.xlabel("x")
    plt.ylabel("np.sin(2.5 * np.pi * X*X)")
    plt.legend()
    plt.title("Simple Linear Regression. Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()
