import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi

INPUT_PATH = "C:/Users/omrys/Git/IML.HUJI/datasets/"
OUTPUT_PATH = "C:/Users/omrys/Git/IML.HUJI/exercises/ex3_graphs/"
ITERATIONS = 1000


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the
    linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for f, n in [("linearly_separable.npy", "Linearly Separable"),
                 ("linearly_inseparable.npy", "Linearly Inseparable")]:
        # Load dataset
        X, y = load_dataset(INPUT_PATH + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        iterations = [*range(1, ITERATIONS + 1)]

        def callback_func(fit: Perceptron, xi: np.ndarray, yi: int):
            losses.append(fit._loss(X, y))

        perceptron = Perceptron(max_iter=ITERATIONS, callback=callback_func)
        perceptron.fit(X, y)
        iterations = iterations[:len(losses)]

        # Plot figure of loss as function of fitting iteration
        fig = px.line(x=iterations, y=losses, title=n,
                      labels={'x': "Iterations", 'y': "Loss"})
        fig.write_image(OUTPUT_PATH + n + ".jpeg")
        # fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f, n in [("gaussian1.npy", "Gaussian1 dataset"),
                 ("gaussian2.npy", "Gaussian2 dataset")]:
        # Load dataset
        X, y = load_dataset(INPUT_PATH + f)

        # Fit models and predict over training set
        gnb = GaussianNaiveBayes()
        gnb._fit(X, y)
        y_gnb = gnb._predict(X)

        lda = LDA()
        lda._fit(X, y)
        y_lda = lda._predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes
        # predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot
        # titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        acc_gnb = accuracy(y, y_gnb)
        acc_lda = accuracy(y, y_lda)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("GNB, accuracy: " + str(acc_gnb),
                                            "LDA, accuracy: " + str(acc_lda)))

        # Add traces for data-points setting symbols and colors
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                       marker=dict(color=y_gnb, symbol=y)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers', showlegend=False,
                       marker=dict(color=y_lda, symbol=y)),
            row=1, col=2,
        )

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_trace(
            go.Scatter(x=gnb.mu_[:, 0], y=gnb.mu_[:, 1], mode='markers',
                       marker=dict(color='black', symbol='cross'),
                       showlegend=False),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode='markers',
                       marker=dict(color='black', symbol='cross'),
                       showlegend=False),
            row=1, col=2
        )

        # Add ellipses depicting the covariances of the fitted Gaussians
        for mu, var in zip(gnb.mu_, gnb.vars_):
            fig.add_trace(get_ellipse(mu, np.diag(var)),
                          row=1, col=1)
        for mu in lda.mu_:
            fig.add_trace(get_ellipse(mu, lda.cov_),
                          row=1, col=2)

        fig.update_layout(height=600, width=800,
                          title_text=n)
        fig.write_image(OUTPUT_PATH + n + ".jpeg")


def quiz():
    x=np.array([0,1,2,3,4,5,6,7])
    y=np.array([0,0,1,1,1,1,2,2])
    gnb = GaussianNaiveBayes()

    x = np.array([[1,1], [1,2], [2,3], [2,4], [3,3], [3,4]])
    y = np.array([0, 0, 1, 1, 1, 1])
    gnb._fit(x, y)
    print(gnb.vars_)
    # run_perceptron()



if __name__ == '__main__':
    np.random.seed(0)
    # run_perceptron()
    # compare_gaussian_classifiers()
    quiz()
