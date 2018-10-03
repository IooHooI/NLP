from sklearn.base import BaseEstimator, TransformerMixin

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical


class SeqTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, max_len=75):
        self.max_len = max_len
        self.unique_words = []
        self.unique_tags = []
        self.word2idx = {}
        self.tag2idx = {}
        self.n_words = len(self.unique_words)
        self.n_tags = len(self.unique_tags)

    def fit(self, X, y=None, **fit_params):
        self.unique_words = list(set([word['lemma'] for sentence in X for word in sentence]))
        self.unique_words.append("ENDPAD")
        self.unique_tags = list(set([word for sentence in y for word in sentence]))
        self.word2idx = {w: i + 1 for i, w in enumerate(self.unique_words)}
        self.tag2idx = {t: i for i, t in enumerate(self.unique_tags)}
        self.n_words = len(self.unique_words)
        self.n_tags = len(self.unique_tags)

    def transform(self, X, y=None, **fit_params):
        X = [[self.word2idx[w['lemma']] for w in s] for s in X]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.n_words - 1)

        y = [[self.tag2idx[w] for w in s] for s in y]
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=self.tag2idx["O"])
        y = [to_categorical(i, num_classes=self.n_tags) for i in y]

        return X, y

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **fit_params)
