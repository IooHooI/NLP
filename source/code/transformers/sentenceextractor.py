from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class SentenceExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        X[X.lemma == '.'] = '%'
        X, y = X[self.features].values, X.ner_tag.values
        if isinstance(self.features, (list,)):
            X, y = np.split(X, np.argwhere(X[:, 0] == '%').flatten()), np.split(y, np.argwhere(y == '%').flatten())
        else:
            X, y = np.split(X, np.argwhere(X == '%').flatten()), np.split(y, np.argwhere(y == '%').flatten())
        for i in range(1, max(len(X), len(y))):
            X[i] = X[i][1:]
            y[i] = y[i][1:]
        return X, y

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X, y, **kwargs)
