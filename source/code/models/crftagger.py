from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn_crfsuite import CRF as crf
from seqeval.metrics import f1_score


class CRFTagger(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.estimator = crf(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=False
        )

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        y_pred = self.estimator.predict(X)
        return y_pred

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X, y)
        return f1_score(y, y_pred)
