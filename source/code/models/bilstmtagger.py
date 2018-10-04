from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin

from keras.models import Model
from keras.models import Input

from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras_contrib.layers import CRF as kerascrf

from keras_tqdm import TQDMNotebookCallback

import numpy as np

import os

from sklearn.metrics import f1_score

from source.code.utils.utils import create_sub_folders


class BiLSTMTagger(BaseEstimator, TransformerMixin, ClassifierMixin):

    def __init__(self, checkpoint_dir='./', max_len=75, batch_size=32, epochs=5, validation_split=0.1):
        self.checkpoint_dir = checkpoint_dir
        create_sub_folders(self.checkpoint_dir)
        self.max_len = max_len
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

    def _build_the_model(self):
        self._input = Input(
            shape=(self.max_len,)
        )
        self.model = Embedding(
            input_dim=self.n_words + 1,
            output_dim=20,
            input_length=self.max_len,
            mask_zero=True
        )(self._input)
        self.model = Bidirectional(
            LSTM(
                units=50,
                return_sequences=True,
                recurrent_dropout=0.1
            )
        )(self.model)
        self.model = TimeDistributed(
            Dense(
                50,
                activation="relu"
            )
        )(self.model)
        self.crf = kerascrf(self.n_tags)
        self.out = self.crf(self.model)
        self.model = Model(
            self._input,
            self.out
        )
        self.model.compile(
            optimizer="rmsprop",
            loss=self.crf.loss_function,
            metrics=[self.crf.accuracy]
        )

    def fit(self, X, y):
        self.words = list(set([word for sentence in X for word in sentence]))
        self.words.append("ENDPAD")
        self.words.append("unknown word")
        self.unique_tags = list(set([tag for sentence in y for tag in sentence]))
        self.word2idx = {w: i + 1 for i, w in enumerate(self.words)}
        self.tag2idx = {t: i for i, t in enumerate(self.unique_tags)}
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}

        self.n_words = len(self.words)
        self.n_tags = len(self.unique_tags)
        self._build_the_model()
        self.model.summary()

        X = [[self.word2idx[w] for w in s] for s in X]
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
            verbose=0,
            callbacks=[
                ModelCheckpoint(
                    filepath=os.path.join(self.checkpoint_dir, 'model.h5'),
                    save_best_only=True,
                    verbose=1
                ),
                EarlyStopping(
                    patience=3,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    min_lr=0.0001,
                    mode='auto',
                    verbose=1
                ),
                TQDMNotebookCallback()
            ]
        )

    def predict(self, X, y=None):
        X = [[self.word2idx[w] if w in self.word2idx else self.word2idx['unknown word'] for w in s] for s in X]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.n_words - 1)
        one_hot_predictions = self.model.predict(X, verbose=1)
        out = []
        for pred_i in one_hot_predictions:
            out_i = []
            for p in pred_i:
                p_i = np.argmax(p)
                out_i.append(self.idx2tag[p_i].replace("PAD", "O"))
            out.append(out_i)
        return out

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return f1_score(y, y_pred, average='macro')
