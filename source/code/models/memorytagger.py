from sklearn.base import BaseEstimator, ClassifierMixin
from seqeval.metrics import f1_score


class MemoryTagger(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.memory = {}
        self.tags = []

    def fit(self, X, y):
        X_cp = [tag[2] for sentence in X for tag in sentence]
        y_cp = [tag for sentence in y for tag in sentence]
        voc = {}
        for x, t in zip(X_cp, y_cp):
            if t not in self.tags:
                self.tags.append(t)
            if x in voc:
                if t in voc[x]:
                    voc[x][t] += 1
                else:
                    voc[x][t] = 1
            else:
                voc[x] = {t: 1}
        for k, d in voc.items():
            self.memory[k] = max(d, key=d.get)
        return self

    def predict(self, X, y=None):
        return [[self.memory.get(tag[2], 'O') for tag in sentence] for sentence in X]

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return f1_score(y, y_pred)
