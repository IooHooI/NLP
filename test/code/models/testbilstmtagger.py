from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from keras.models import Model
from keras.models import Input

from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras_contrib.layers import CRF as kerascrf

import numpy as np

from sklearn.metrics import f1_score


class BiLSTMTagger(BaseEstimator, TransformerMixin, ClassifierMixin):

    def __init__(self, max_len=75, batch_size=32, epochs=5, validation_split=0.1):
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

    def _build_the_model(self):
        self._input = Input(shape=(self.max_len,))

        self.model = Embedding(input_dim=self.n_words + 1, output_dim=20, input_length=self.max_len, mask_zero=True)(self._input)
        self.model = Bidirectional(LSTM(units=50, return_sequences=True, recurrent_dropout=0.1))(self.model)
        self.model = TimeDistributed(Dense(50, activation="relu"))(self.model)

        self.crf = kerascrf(self.n_tags)
        self.out = self.crf(self.model)

        self.model = Model(self._input, self.out)
        self.model.compile(optimizer="rmsprop", loss=kerascrf.loss_function, metrics=[kerascrf.accuracy])

    def fit(self, X, y):
        self.words = list(set([word['lemma'] for sentence in X for word in sentence]))
        self.words.append("ENDPAD")
        self.unique_tags = list(set([word for sentence in y for word in sentence]))
        self.word2idx = {w: i + 1 for i, w in enumerate(self.words)}
        self.tag2idx = {t: i for i, t in enumerate(self.unique_tags)}

        self.n_words = len(self.words)
        self.n_tags = len(self.unique_tags)
        self._build_the_model()
        self.model.summary()

        X = [[self.word2idx[w['lemma']] for w in s] for s in X]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.n_words - 1)

        y = [[self.tag2idx[w] for w in s] for s in y]
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=self.tag2idx["O"])
        y = [to_categorical(i, num_classes=self.n_tags) for i in y]

        self.model.fit(
            X,
            np.array(y),
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.validation_split,
            verbose=1
        )

    def predict(self, X, y=None):
        X = [[self.word2idx[w['lemma']] for w in s] for s in X]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.n_words - 1)
        return self.model.predict(X, verbose=1)

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro')
