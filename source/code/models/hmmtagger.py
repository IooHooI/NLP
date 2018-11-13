from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from tqdm.autonotebook import tqdm
from functools import reduce
import numpy as np


class HMMTagger(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=0.01, features=None):
        # if features is None:
        #     self.features = ['word']
        # else:
        #     self.features = features
        self.features = features

        self.alpha = alpha

        self.words_binarizers = None

        self.word_2_idx_maps = None

        self.idx_2_word_maps = None

        self.tags_binarizer = None

        self.tag2idx = None

        self.idx2tag = None

        self.Pi = None

        self.A = None

        self.B = None

    def fit(self, X, y):
        X_cp = X.copy()
        self.tag2idx, self.idx2tag = self._get_states_mapping_from(y)
        if self.features is None:
            word_2_idx_maps, idx_2_word_maps = self._get_states_mapping_from(X_cp)
            self.word_2_idx_maps = [word_2_idx_maps]
            self.idx_2_word_maps = [idx_2_word_maps]
        else:
            X_cp = self._split_sentences_by_features(X_cp)
            mappings = [self._get_states_mapping_from(sentences) for sentences in X_cp]
            self.word_2_idx_maps = [mappings[i][0] for i in range(len(self.features))]
            self.idx_2_word_maps = [mappings[i][1] for i in range(len(self.features))]
        self._calculate_pi(y)
        self._calculate_a(y)
        self._calculate_b(X_cp, y)
        return self

    def predict(self, X, y=None):
        X_cp = X.copy()
        y_pred = []
        for sentence in tqdm(X_cp, desc='PREDICTIONS CALCULATION: '):
            path, _, _ = self._viterbi(sentence)
            y_pred.append([self.idx2tag[step] for step in path])
        return y_pred

    def _encode_sentences(self, sentences, dictionary):
        encoded_sentences = []
        for sentence in tqdm(sentences, desc='FLATTEN SENTENCES: '):
            encoded_sentences.extend(self._encode_sentence(sentence))
        encoded_sentences = np.array(list(map(lambda x: dictionary[x], tqdm(encoded_sentences, desc='STATE TO IDX: '))))
        return encoded_sentences

    @staticmethod
    def _encode_sentence(sentence):
        encoded_sentence = []
        for word in sentence:
            encoded_sentence.append(word)
        return encoded_sentence

    def _split_sentences_by_features(self, sentences):
        sentences_split = [
            [sentence[:, i] for sentence in sentences] for i in range(len(self.features))
        ]
        return sentences_split

    @staticmethod
    def _get_states_mapping_from(sentences):
        states = list(set([tag for sentence in sentences for tag in sentence]))
        state2idx = {t: i for i, t in enumerate(states)}
        idx2state = {i: w for w, i in state2idx.items()}

        return state2idx, idx2state

    def _calculate_pi(self, y):
        self.Pi = np.zeros(len(self.tag2idx.keys()))
        for sentence in tqdm(y, desc='INITIAL PROBS CALCULATION: '):
            if len(sentence) > 0:
                self.Pi[self.tag2idx[sentence[0]]] += 1
        self.Pi = (self.Pi + self.alpha) / (len(y) + self.alpha)

    def _calculate_a(self, y):
        self.A = self._calculate_table(
            y,
            np.roll(y, 1),
            self.tag2idx,
            self.tag2idx
        )

    def _calculate_table(self, X, y, X_map, y_map):
        binarizer = LabelBinarizer(sparse_output=True)
        current_open_states_sequence = self._encode_sentences(X, X_map)
        current_hidden_states_sequence = self._encode_sentences(y, y_map)
        nb = MultinomialNB(alpha=self.alpha)
        nb.fit(
            binarizer.fit_transform(
                current_open_states_sequence.reshape(-1, 1)
            ),
            current_hidden_states_sequence
        )
        return np.exp(nb.feature_log_prob_)

    def _calculate_b(self, X, y):
        if self.features is None:
            self.B = [self._calculate_table(
                X,
                y,
                self.word_2_idx_maps[0],
                self.tag2idx
            )]
        else:
            self.B = [self._calculate_table(
                X[i],
                y,
                self.word_2_idx_maps[i],
                self.tag2idx
            ) for i in range(len(self.features))]

    def _viterbi(self, obs):
        n_states = np.shape(self.A)[0]
        T = np.shape(obs)[0]
        path = np.zeros(T)
        delta = np.zeros((n_states, T))
        phi = np.zeros((n_states, T))
        if self.features is None:
            b_calc_res = self.B[0][:, self.word_2_idx_maps[0][obs[0]]] if obs[0] in self.word_2_idx_maps[0] else 0.5
            delta[:, 0] = self.Pi * b_calc_res
            for t in range(1, T):
                for s in range(n_states):
                    b_calc_res = self.B[0][s, self.word_2_idx_maps[0][obs[t]]] if obs[t] in self.word_2_idx_maps[0] else 0.5
                    delta[s, t] = np.max(delta[:, t - 1] * self.A[:, s]) * b_calc_res
                    phi[s, t] = np.argmax(delta[:, t - 1] * self.A[:, s])
            path[T - 1] = np.argmax(delta[:, T - 1])
            for t in range(T - 2, -1, -1):
                path[t] = phi[int(path[t + 1]), int(t + 1)]
            return path, delta, phi
        else:
            b_calc_res = reduce(
                (lambda x, y: x * y),
                [
                    self.B[
                        i
                    ][
                        :,
                        self.word_2_idx_maps[i][obs[0][i]]
                    ] if obs[0][i] in self.word_2_idx_maps[i] else 0.5 for i in range(len(self.features))
                ]
            )
            delta[:, 0] = self.Pi * b_calc_res
            for t in range(1, T):
                for s in range(n_states):
                    b_calc_res = reduce(
                        (lambda x, y: x * y),
                        [
                            self.B[
                                i
                            ][
                                s,
                                self.word_2_idx_maps[i][obs[t][i]]
                            ] if obs[t][i] in self.word_2_idx_maps[i] else 0.5 for i in range(len(self.features))
                        ]
                    )
                    delta[s, t] = np.max(delta[:, t - 1] * self.A[:, s]) * b_calc_res
                    phi[s, t] = np.argmax(delta[:, t - 1] * self.A[:, s])
            path[T - 1] = np.argmax(delta[:, T - 1])
            for t in range(T - 2, -1, -1):
                path[t] = phi[int(path[t + 1]), int(t + 1)]
            return path, delta, phi

    def score(self, X, y, sample_weight=None):
        pass
