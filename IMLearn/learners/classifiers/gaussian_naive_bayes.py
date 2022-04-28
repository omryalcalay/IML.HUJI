from typing import NoReturn

from ...base import BaseEstimator
import numpy as np

class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        pi = []
        mean_vecs = []
        vars_vecs = []
        for label in self.classes_:
            pi.append(y[y == label].size / y.size)
            mean_vecs.append(np.mean(X[y == label], axis=0))
            vars_vecs.append(np.var(X[y == label], axis=0))
        self.pi_ = np.array(pi)
        self.mu_ = np.array(mean_vecs)
        self.vars_ = np.array(vars_vecs)
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
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        y_pred = []
        for sample in X:
            yi = []
            for i in range(len(self.classes_)):
                mu_cur = self.mu_[i].reshape((len(self.mu_[i]), 1))
                pi_cur = self.pi_[i]
                cov_cur = np.diag(self.vars_[i])
                cov_inv_cur = np.linalg.inv(cov_cur)
                a = (cov_inv_cur @ mu_cur).T @ sample
                b = np.diag(
                    np.log(pi_cur / np.sqrt(np.linalg.norm(cov_cur))) - .5 * (
                            (mu_cur.T @ cov_inv_cur) @ mu_cur
                    )
                )
                c = -.5 * sample.T @ cov_inv_cur @ sample
                yi.append(a + b + c)
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
