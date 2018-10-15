from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class SentenceExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X, y=None, **fit_params):
        X[X.lemma == '.'] = '%'
        X, y = X.lemma.values, X.ner_tag.values
        X, y = np.split(X, np.argwhere(X == '%').flatten()), np.split(y, np.argwhere(y == '%').flatten())
        for i in range(1, max(len(X), len(y))):
            X[i] = X[i][1:]
            y[i] = y[i][1:]
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)
