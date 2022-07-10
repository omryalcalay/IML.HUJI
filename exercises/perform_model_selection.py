from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots

INPUT_PATH = "C:/Users/omrys/Git/IML.HUJI/"
OUTPUT_PATH = "C:/Users/omrys/Git/IML.HUJI/exercises/ex5_graphs/"


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.linspace(-1.2, 2, n_samples)
    np.random.shuffle(X)
    def f(x): return (x + 3)*(x + 2)*(x + 1)*(x - 1)*(x - 2)
    noiseless = np.vectorize(f)(X)
    eps = np.random.normal(0, noise, size=n_samples)
    y = noiseless + eps
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), 2/3)
    train_X, train_y, test_X, test_y =\
        np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)
    fig = go.Figure([
        go.Scatter(x=X, y=noiseless, name="Noiseless", mode="markers"),
        go.Scatter(x=train_X[:, 0], y=train_y, name="Train", mode="markers"),
        go.Scatter(x=test_X[:, 0], y=test_y, name="Test", mode="markers")])
    # fig.show()
    fig.write_image(OUTPUT_PATH + "Q1_noise" + str(noise) + ".jpeg")

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    k_range = np.arange(11)
    train_errs = []
    valid_errs = []
    for k in k_range:
        train, valid = cross_validate(PolynomialFitting(k), train_X, train_y, mean_square_error)
        train_errs.append(train)
        valid_errs.append(valid)
    fig = go.Figure([
        go.Scatter(x=k_range, y=train_errs, name="Train errors"),
        go.Scatter(x=k_range, y=valid_errs, name="Validation errors")],
        layout=go.Layout(
            title="Errors as a function of hyperparameter K",
            xaxis=dict(title="K - polynom rank"))
    )
    # fig.show()
    fig.write_image(OUTPUT_PATH + "Q2_noise" + str(noise) + ".jpeg")

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    k_star = k_range[np.argmin(valid_errs)]
    poly = PolynomialFitting(k_star)
    poly.fit(X.reshape((X.shape[0], 1)), y)
    print("K* is: " + str(k_star) + ", and it's test error: "
          + str(round(poly.loss(test_X[:, 0], test_y), 2)))


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    from sklearn import datasets
    data = datasets.load_diabetes()
    X, y = data.data, data.target
    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X), pd.Series(y), n_samples/y.size)
    train_X, train_y, test_X, test_y = \
        np.array(train_X), np.array(train_y), np.array(test_X), np.array(
            test_y)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lam_range = [np.linspace(0.01, 0.5, n_evaluations),
                 np.linspace(0.01, 3, n_evaluations)]
    train_errs = [[], []]
    valid_errs = [[], []]
    models = [RidgeRegression, Lasso]
    model_names = ["Ridge Regression", "Lasso"]
    for i, model in enumerate(models):
        for lam in lam_range[i]:
            train, valid = cross_validate(model(lam), train_X, train_y, mean_square_error)
            train_errs[i].append(train)
            valid_errs[i].append(valid)
        fig = go.Figure([
            go.Scatter(x=lam_range[i], y=train_errs[i], name="Train errors"),
            go.Scatter(x=lam_range[i], y=valid_errs[i], name="Validation errors")],
            layout=go.Layout(
                title=model_names[i] + " errors as a function of hyperparameter Lambda",
                xaxis=dict(title="Lambda - regularization parameter"))
        )
        # fig.show()
        fig.write_image(OUTPUT_PATH + "Q7_" + model_names[i] + ".jpeg")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    lam_star = []*len(model_names)
    for i, name in enumerate(model_names):
        lam_star.append((lam_range[i])[np.argmin(valid_errs[i])])
        print("For model " + model_names[i] +
              " the best regularization parameter is " + str(lam_star[i]))
    models.append(LinearRegression)
    model_names.append("Least Squares")
    for i, model in enumerate(models):
        if i == 2:
            initialized = model()
        else:
            initialized = model(lam_star[i])
        initialized.fit(train_X, train_y)
        if i == 1:
            loss = mean_square_error(test_y, initialized.predict(test_X))
        else:
            loss = initialized.loss(test_X, test_y)
        print("For model " + model_names[i] +
              " the test error is " + str(loss))


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
