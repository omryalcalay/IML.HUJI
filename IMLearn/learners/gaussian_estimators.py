from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

from numpy.linalg import inv, det, slogdet

NO_OF_SAMPLES = 1000
MU = 10
VAR = 1


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """

        self.mu_ = np.mean(X)
        ddof = 0 if self.biased_ else 1
        self.var_ = np.var(X, ddof=ddof)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        return np.fromiter((self.__f(x) for x in X), X.dtype)

    def __f(self, x):
        return (1 / (np.sqrt(2 * np.pi * self.var_))) \
               * np.exp(-(1 / (2 * self.var_)) * (x - self.mu_) ** 2)

    @staticmethod
    def log_likelihood(mu: float, variance: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        variance : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        m = X.size
        return -m/2 * np.log(2*np.pi*variance) - ((1/2*variance) * ((X-mu)**2).sum())


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: ndarray of shape (n_features,)
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.fit`
            function.

        cov_: ndarray of shape (n_features, n_features)
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.fit`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        self : returns an instance of self

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = np.cov(X, rowvar=False)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        return np.fromiter((self.__f(x) for x in X), X.dtype)

    def __f(self, x):
        m = x.size
        cov_det = np.linalg.det(self.cov_)
        c = 1/np.sqrt(((2*np.pi)**m) * cov_det)
        x_ = x - self.mu_
        return c*np.exp(-(1/2) * np.multiply(np.multiply(x_.T, cov_det)), x_)


    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : ndarray of shape (n_features,)
            Expectation of Gaussian
        cov : ndarray of shape (n_features, n_features)
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, n_features)
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated over all input data and under given parameters of Gaussian
        """
        sums = np.zeros((X[0].size, X[0].size))
        for x in X:
            x_mu = np.add(x, (-1)*mu)
            sums = np.add(sums, np.multiply(x_mu, x_mu.T))
        trace = -np.trace(np.multiply(sums, np.linalg.inv(cov)))/2
        det_ = -X.size / 2 * np.log(np.linalg.det(cov))
        return trace + det_
        # row, col = X.shape
        # l = col * np.log(2*np.pi) + np.log(det(cov))
        # x_ = X - mu
        # return -(row * l + (np.multiply(np.multiply(x_.T, inv(cov)), x_)).sum())/2
