from sklearn.base import BaseEstimator, ClassifierMixin


class MemoryTagger(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.memory = {}
        self.tags = []

    def fit(self, X, y):
        voc = {}
        for x, t in zip(X, y):
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
        return [self.memory.get(x, 'O') for x in X]

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
