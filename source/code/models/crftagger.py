from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class CRFTagger(BaseEstimator, ClassifierMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        return self

    def predict(self, X, y=None):
        pass

    def score(self, X, y, sample_weight=None):
        pass
