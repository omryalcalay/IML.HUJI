import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

import plotly.graph_objects as go

from utils import custom

OUTPUT_PATH = "C:/Users/omrys/Git/IML.HUJI/exercises/ex6_graphs/"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's
     value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding
        the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []
    ts = []

    def callback(solver, ws, val, grad, t, eta, delta):
        values.append(val[0])
        weights.append(ws)
        ts.append(t)

    return callback, values, weights, ts


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = [L1, L2]
    values = []
    weights = []
    iterations = []
    min_loss = []
    for i, module in enumerate(modules):
        values.append([])
        weights.append([])
        iterations.append([])
        min_loss.append([])
        for j, eta in enumerate(etas):
            callback, vals, weight, ts = get_gd_state_recorder_callback()
            lr = FixedLR(eta)
            gd = GradientDescent(learning_rate=lr, callback=callback)
            gd.fit(module(init), None, None)
            values[i].append(vals)
            weights[i].append(np.array(weight))
            iterations[i].append(ts)
            min_loss[i].append(np.min(vals))
            fig = plot_descent_path(module=module, descent_path=weights[i][j],
                                    title="for module L" + str(i+1)
                                          + ", eta=" + str(eta))
            # fig.show()
            fig.write_image(
                OUTPUT_PATH + "Q1_L" + str(i+1) + "_eta" + str(eta) + ".jpeg")
        fig = go.Figure([go.Scatter(
            x=iterations[i][k], y=values[i][k],
            name="eta=" + str(etas[k])) for k in range(len(etas))],
            layout=go.Layout(
                title="Convergence rate for L" + str(i+1) +
                      " with fixed learning rate",
                xaxis=dict(title="Iteration"))
        )
        # fig.show()
        fig.write_image(OUTPUT_PATH + "Q3_L" + str(i + 1) + ".jpeg")
        print("lowest loss for L" + str(i+1) + " is " + str(np.min(min_loss[i])))


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the
    # exponentially decaying learning rate
    values = []
    weights = []
    iterations = []
    min_loss = []
    for gamma in gammas:
        callback, vals, weight, ts = get_gd_state_recorder_callback()
        lr = ExponentialLR(base_lr=eta, decay_rate=gamma)
        gd = GradientDescent(learning_rate=lr, callback=callback)
        gd.fit(L1(init), None, None)
        values.append(vals)
        weights.append(np.array(weight))
        iterations.append(ts)
        min_loss.append(np.min(vals))

    # Plot algorithm's convergence for the different values of gamma
    fig = go.Figure([go.Scatter(
        x=iterations[k], y=values[k],
        name="gamma=" + str(gammas[k])) for k in range(len(gammas))],
        layout=go.Layout(
            title="Convergence rate for L1 with decay rate",
            xaxis=dict(title="Iteration"))
    )
    # fig.show()
    fig.write_image(OUTPUT_PATH + "Q5.jpeg")

    # Q6
    print("lowest loss is " + str(np.min(min_loss)))

    # Plot descent path for gamma=0.95
    modules = [L1, L2]
    callback, vals, weight, ts = get_gd_state_recorder_callback()
    lr = ExponentialLR(base_lr=eta, decay_rate=0.95)
    gd = GradientDescent(learning_rate=lr, callback=callback)
    gd.fit(modules[1](init), None, None)
    weights_2 = [weights[1], np.array(weight)]

    for i, module in enumerate(modules):
        fig = plot_descent_path(module=module, descent_path=weights_2[i],
                                title="Trajectory for module L" + str(i + 1))
        # fig.show()
        fig.write_image(OUTPUT_PATH + "Q7_L" + str(i + 1) + ".jpeg")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = \
        np.array(X_train), np.array(y_train), np.array(X_test), np.array(
            y_test)
    log_reg = LogisticRegression(solver=GradientDescent(max_iter=20000))
    log_reg.fit(X_train, y_train)

    # Plotting convergence rate of logistic regression over SA heart disease data
    from sklearn.metrics import roc_curve, auc
    y_prob = log_reg.predict_proba(X_train)
    fpr, tpr, thresholds = roc_curve(y_train, y_prob)
    fig = go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=custom[-1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))
    # fig.show()
    fig.write_image(OUTPUT_PATH + "Q8.jpeg")
    # Q9
    alpha_star = thresholds[np.argmax(tpr - fpr)]
    log_reg = LogisticRegression(alpha=alpha_star)
    log_reg.fit(X_test, y_test)
    print("alpha* is " + str(alpha_star) + " and it's test error is " +
          str(log_reg.loss(X_test, y_test)))

    # Fitting l1- and l2-regularized logistic regression models, using
    # cross-validation to specify values
    # of regularization parameter
    regs = ["l1", "l2"]
    lams = [.001, .002, .005, .01, .02, .05, .1]
    lam_star = [0, 0]
    train_errs = [[], []]
    valid_errs = [[], []]
    for i, reg in enumerate(regs):
        for lam in lams:
            log_reg = LogisticRegression(penalty=reg, lam=lam)
            train, valid = cross_validate(log_reg, X_train, y_train,
                                          misclassification_error)
            train_errs[i].append(train)
            valid_errs[i].append(valid)
        lam_star[i] = lams[np.argmin(valid_errs[i])]
        log_reg = LogisticRegression(penalty=reg, lam=lam_star[i])
        log_reg.fit(X=X_train, y=y_train)
        print("for module " + reg + " lam* is " + str(lam_star[i]) +
              " and it's test error is " +
              str(log_reg.loss(X=X_test, y=y_test)))


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
