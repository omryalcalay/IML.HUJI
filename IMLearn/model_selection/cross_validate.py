from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    lim_i = int(np.floor(X.shape[0] / cv))
    train_errs = []
    valid_errs = []
    for i in range(cv):
        valid_X = X[i * lim_i:(i + 1) * lim_i, :]
        train_X = np.concatenate([X[:i * lim_i, :], X[(i + 1) * lim_i:, :]])
        valid_y = y[i * lim_i:(i + 1) * lim_i]
        train_y = np.concatenate([y[:i * lim_i], y[(i + 1) * lim_i:]])
        estimator.fit(train_X, train_y)
        if X.shape == (X.shape[0], 1):
            train_X = train_X[:, 0]
            valid_X = valid_X[:, 0]
        train_pred = estimator.predict(train_X)
        valid_pred = estimator.predict(valid_X)
        train_errs.append(scoring(train_y, train_pred))
        valid_errs.append(scoring(valid_y, valid_pred))
    return np.average(train_errs), np.average(valid_errs)

    # ids = np.arange(X.shape[0])
    #
    # # Randomly split samples into `cv` folds
    # folds = np.array_split(ids, cv)
    #
    # train_score, validation_score = .0, .0
    # for fold_ids in folds:
    #     train_msk = ~np.isin(ids, fold_ids)
    #     fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])
    #
    #     train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
    #     validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))
    #
    # return train_score / cv, validation_score / cv
