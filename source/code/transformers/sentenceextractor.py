from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class SentenceExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None, **fit_params):
        pass

    def transform(self, X, y=None, **fit_params):
        X[X.lemma == '.'] = '%'
        X, y = np.array([np.array(s.strip().split('|')) for s in '|'.join(X.lemma.values).strip().split('%')]), \
               np.array([np.array(s.strip().split('|')) for s in '|'.join(X.ner_tag.values).strip().split('%')])
        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)
