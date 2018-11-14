from sklearn.base import BaseEstimator, TransformerMixin
from tqdm.autonotebook import tqdm


class CRFTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None, **kwargs):
        return self

    def transform(self, X, y=None, **kwargs):
        X = [[dict(zip(self.features, word)) for word in sentence] for sentence in tqdm(X, desc='CRF TRANSFORMATION: ')]
        return X

    def fit_transform(self, X, y=None, **kwargs):
        self.fit(X, y, **kwargs)
        return self.transform(X, y, **kwargs)
