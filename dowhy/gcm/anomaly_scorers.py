from typing import Optional

import numpy as np
from statsmodels.robust import mad

from dowhy.gcm.constant import EPS
from dowhy.gcm.anomaly_scorer import AnomalyScorer
from dowhy.gcm.density_estimator import DensityEstimator
from dowhy.gcm.util.general import shape_into_2d


class MedianCDFQuantileScorer(AnomalyScorer):

    def __init__(self):
        self.__distribution_samples = None

    def fit(self, X: np.ndarray) -> None:
        if (X.ndim == 2 and X.shape[1] > 1) or X.ndim > 2:
            raise ValueError('The MedianCDFQuantileScorer currently only supports one-dimensional data!')

        self.__distribution_samples = X.reshape(-1)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.__distribution_samples is None:
            raise ValueError("Scorer has not been fitted!")

        X = shape_into_2d(X)

        equal_samples = np.sum(X == self.__distribution_samples, axis=1)
        greater_samples = np.sum(X > self.__distribution_samples, axis=1) + equal_samples / 2
        smaller_samples = np.sum(X < self.__distribution_samples, axis=1) + equal_samples / 2

        return 1 - 2 * np.amin(np.vstack([greater_samples, smaller_samples]), axis=0) \
               / self.__distribution_samples.shape[0]


class RescaledMedianCDFQuantileScorer(AnomalyScorer):

    def __init__(self):
        self.__scorer = MedianCDFQuantileScorer()

    def fit(self, X: np.ndarray) -> None:
        self.__scorer.fit(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        scores = 1 - self.__scorer.score(X)
        scores[scores == 0] = EPS

        return -np.log(scores)


class ITAnomalyScorer(AnomalyScorer):
    def __init__(self, anomaly_scorer: AnomalyScorer):
        self.__anomaly_scorer = anomaly_scorer
        self.__distribution_samples = None
        self.__scores_of_distribution_samples = None

    def fit(self, X: np.ndarray) -> None:
        self.__distribution_samples = shape_into_2d(X)
        self.__anomaly_scorer.fit(self.__distribution_samples)
        self.__scores_of_distribution_samples = self.__anomaly_scorer.score(self.__distribution_samples).reshape(-1)

    def score(self, X: np.ndarray) -> np.ndarray:
        X = shape_into_2d(X)
        scores_of_samples_to_score = self.__anomaly_scorer.score(X).reshape(-1, 1)
        return -np.log((np.sum(self.__scores_of_distribution_samples >= scores_of_samples_to_score, axis=1) + 0.5)
                       / (self.__scores_of_distribution_samples.shape[0] + 0.5))


class MeanDeviationScorer(AnomalyScorer):

    def __init__(self):
        self.__mean = None
        self.__std = None

    def fit(self, X: np.ndarray) -> None:
        self.__mean = np.mean(X)
        self.__std = np.std(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.__mean is None or self.__std is None:
            raise ValueError("Scorer has not been fitted!")

        return abs(X - self.__mean) / self.__std


class MedianDeviationScorer(AnomalyScorer):

    def __init__(self):
        self.__median = None
        self.__mad = None

    def fit(self, X: np.ndarray) -> None:
        self.__median = np.median(X)
        self.__mad = mad(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.__median is None or self.__mad is None:
            raise ValueError("Scorer has not been fitted!")

        return abs(X - self.__median) / self.__mad


class InverseDensityScorer(AnomalyScorer):
    """
    Estimates an anomaly score based on 1 / p(x), where x is the data to score. The density value p(x) is estimated
    using the given density estimator. If None is given, a Gaussian mixture model is used by default.

    Note: The given density estimator needs to support the data types, i.e. if the data has categorical values, the
    density estimator needs to be able to handle that. The default Gaussian model can only handle numeric data.

    Note: If the density p(x) is 0, a nan or inf could be returned.
    """

    def __init__(self, density_estimator: Optional[DensityEstimator] = None):
        if density_estimator is None:
            from causality.density_estimators import GaussianMixtureDensityEstimator
            density_estimator = GaussianMixtureDensityEstimator()
        self.__density_estimator = density_estimator
        self.__fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.__density_estimator.fit(X)
        self.__fitted = True

    def score(self, X: np.ndarray) -> np.ndarray:
        if not self.__fitted:
            raise ValueError("Scorer has not been fitted!")

        return 1 / self.__density_estimator.estimate_density(X)
