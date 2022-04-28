from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        mean_vecs = []
        pi = []
        for label in self.classes_:
            mean_vecs.append(np.mean(X[y == label], axis=0))
            pi.append(y[y == label].size / y.size)
        self.mu_ = np.array(mean_vecs)
        self.pi_ = np.array(pi)
        n_features = X.shape[1]
        self.cov_ = np.zeros((n_features, n_features))
        for label, mean_vec in zip(self.classes_, mean_vecs):
            label_cov = np.zeros((n_features, n_features))
            for sample in X[y == label]:
                sample, mean_vec = sample.reshape(n_features, 1),\
                                   mean_vec.reshape(n_features, 1)
                label_cov += (sample - mean_vec).dot((sample - mean_vec).T)
            self.cov_ += label_cov
        self.cov_ /= (y.size - self.classes_.size)
        self._cov_inv = inv(self.cov_)
        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        y_pred = []
        for sample_likelihood in self.likelihood(X):
            yi = self.classes_[np.argmax(sample_likelihood)]
            y_pred.append(yi)
        return np.array(y_pred)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")
        y_pred = []
        a = self._cov_inv @ self.mu_.T
        b = np.diag(
            np.log(self.pi_) - .5 * ((self.mu_ @ self._cov_inv) @ self.mu_.T))
        for sample in X:
            yi = a.T @ sample + b
            y_pred.append(yi)
        return np.array(y_pred)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))
