from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class VotingModel(BaseEstimator, ClassifierMixin):
    """A voting ensemble model that aggregates predictions from multiple models"""

    def __init__(self, estimators: list[BaseEstimator]):
        """Initialize the VotingModel with a list of base estimators

        Args:
            - estimators (list): base estimators
        """
        super().__init__()

        self.estimators = estimators

    def fit(self, X, y=None):
        """Fit the model to the training data

        Args:
            - X: Input features
            - y: Target labels
        """
        return self

    def predict(self, X):
        """Predict class labels for samples

        Args:
            - X: Input features
        Returns:
            - np.ndarray: Predicted class labels
        """
        y_preds = [v.predict(X) for v in self.estimators]

        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        """Predict class probabilities for samples

        Args:
            - X: Input features
        Returns:
            - np.ndarray: Predicted class probabilities
        """
        y_preds = [v.predict_proba(X) for v in self.estimators]

        return np.mean(y_preds, axis=0)
