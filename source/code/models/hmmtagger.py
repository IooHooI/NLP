from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
import numpy as np


class HMMTagger(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=0.01, features=None):
        self.tags_binarizer = LabelBinarizer()
        self.features = features
        self.alpha = alpha
        self.tag2idx = None
        self.idx2tag = None
        self.pi = None
        self.A = None
        self.B = None
        self.sentence_begin_tag = '_SB_'
        self.sentence_end_tag = '_SE_'

    def fit(self, X, y):
        self._calculate_pi(y)
        self._calculate_a(y)
        self._calculate_b(X, y)
        return self

    def predict(self, X, y=None):
        pass

    def _calculate_pi(self, y):
        states = list(set([tag for sentence in y for tag in sentence]))
        states.append(self.sentence_begin_tag)
        states.append(self.sentence_end_tag)
        self.tag2idx = {t: i for i, t in enumerate(states)}
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}

    def _calculate_a(self, y):
        current_states_sequence = []
        for sentence in y:
            current_states_sequence.append(self.sentence_begin_tag)
            for tag in sentence:
                current_states_sequence.append(tag)
            current_states_sequence.append(self.sentence_end_tag)
        previous_states_sequence = np.roll(current_states_sequence, 1)

        current_states_sequence = list(map(lambda x: self.tag2idx[x], current_states_sequence))
        previous_states_sequence = list(map(lambda x: self.tag2idx[x], previous_states_sequence))

        current_states_sequence = np.array(current_states_sequence)
        previous_states_sequence = np.array(previous_states_sequence)

        nb = MultinomialNB(alpha=self.alpha)
        nb.fit(
            self.tags_binarizer.fit_transform(previous_states_sequence.reshape(-1, 1)),
            current_states_sequence
        )
        self.A = np.exp(nb.feature_log_prob_)

    def _calculate_b(self, X, y):
        if self.features is None:
            nb = MultinomialNB(alpha=self.alpha)
            nb.fit(
                X,
                y
            )
            self.B = np.exp(nb.feature_log_prob_)
        else:
            pass
        pass

    @staticmethod
    def _viterbi(self, pi, A, B, obs):
        n_states = np.shape(B)[0]

        T = np.shape(obs)[0]

        path = np.zeros(T)

        delta = np.zeros((n_states, T))

        phi = np.zeros((n_states, T))

        delta[:, 0] = pi * B[:, obs[0]]

        phi[:, 0] = 0

        for t in range(1, T):
            for s in range(n_states):
                delta[s, t] = np.max(delta[:, t - 1] * A[:, s]) * B[s, obs[t]]
                phi[s, t] = np.argmax(delta[:, t - 1] * A[:, s])

        path[T - 1] = np.argmax(delta[:, T - 1])

        for t in range(T - 2, -1, -1):
            path[t] = phi[int(path[t + 1]), int(t + 1)]

        return path, delta, phi

    def score(self, X, y, sample_weight=None):
        pass
