from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelBinarizer
from tqdm.autonotebook import tqdm
import numpy as np


class HMMTagger(BaseEstimator, ClassifierMixin):

    def __init__(self, alpha=0.01, features=None):
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
        self.tag2idx, self.idx2tag = self._get_states_mapping_from(y)
        if self.features is None:
            word_2_idx_maps, idx_2_word_maps = self._get_states_mapping_from(X)
            self.word_2_idx_maps = [word_2_idx_maps]
            self.idx_2_word_maps = [idx_2_word_maps]
        else:
            # TODO: case with several features
            pass
        self._calculate_pi(y)
        self._calculate_a(y)
        self._calculate_b(X, y)
        return self

    def predict(self, X, y=None):
        if self.features is None:
            y_pred = []
            for sentence in tqdm(X, desc='PREDICTIONS CALCULATION: '):
                path, _, _ = self._viterbi(sentence)
                y_pred.append([self.idx2tag[step] for step in path])
            return y_pred
        else:
            pass

    def _encode_sentences(self, sentences, dictionary):
        encoded_sentences = []
        for sentence in tqdm(sentences, desc='FLATTEN SENTENCES: '):
            encoded_sentences.extend(self._encode_sentence(sentence))
        encoded_sentences = np.array(list(map(lambda x: dictionary[x], tqdm(encoded_sentences, desc='STATE TO IDX: '))))
        return encoded_sentences

    def _encode_sentence(self, sentence):
        encoded_sentence = []
        for word in sentence:
            encoded_sentence.append(word)
        return encoded_sentence

    def _get_states_mapping_from(self, sentences):
        states = list(set([tag for sentence in sentences for tag in sentence]))
        state2idx = {t: i for i, t in enumerate(states)}
        idx2state = {i: w for w, i in state2idx.items()}

        return state2idx, idx2state

    def _calculate_pi(self, y):
        self.Pi = np.zeros(len(self.tag2idx.keys()))
        for sentence in tqdm(y, desc='INITIAL PROBS CALCULATION: '):
            if len(sentence) > 0:
                self.Pi[self.tag2idx[sentence[0]]] += 1

    def _calculate_a(self, y):
        self.tags_binarizer = LabelBinarizer(sparse_output=True)
        current_hidden_states_sequence = self._encode_sentences(y, self.tag2idx)
        previous_hidden_states_sequence = np.roll(current_hidden_states_sequence, 1)
        nb = MultinomialNB(alpha=self.alpha)
        nb.fit(
            self.tags_binarizer.fit_transform(
                previous_hidden_states_sequence.reshape(-1, 1)
            ),
            current_hidden_states_sequence
        )
        self.A = np.exp(nb.feature_log_prob_)

    def _calculate_b(self, X, y):
        if self.features is None:
            self.words_binarizers = [LabelBinarizer(sparse_output=True)]
            current_hidden_states_sequence = self._encode_sentences(y, self.tag2idx)
            current_open_states_sequence = self._encode_sentences(X, self.word_2_idx_maps[0])
            nb = MultinomialNB(alpha=self.alpha)
            nb.fit(
                self.words_binarizers[0].fit_transform(
                    current_open_states_sequence.reshape(-1, 1)
                ),
                current_hidden_states_sequence
            )
            self.B = np.exp(nb.feature_log_prob_)
        else:
            self.words_binarizers = [LabelBinarizer()] * len(self.features)

            # TODO: case with several features

    def _viterbi(self, obs):
        n_states = np.shape(self.B)[0]
        T = np.shape(obs)[0]
        path = np.zeros(T)
        delta = np.zeros((n_states, T))
        phi = np.zeros((n_states, T))
        delta[:, 0] = self.Pi * self.B[:, self.word_2_idx_maps[0][obs[0]]] if obs[0] in self.word_2_idx_maps[0] else 0.5
        phi[:, 0] = 0
        for t in range(1, T):
            for s in range(n_states):
                delta[s, t] = np.max(
                    delta[:, t - 1] * self.A[:, s]
                ) * (
                    self.B[s, self.word_2_idx_maps[0][obs[t]]] if obs[t] in self.word_2_idx_maps[0] else 0.5
                )
                phi[s, t] = np.argmax(delta[:, t - 1] * self.A[:, s])
        path[T - 1] = np.argmax(delta[:, T - 1])
        for t in range(T - 2, -1, -1):
            path[t] = phi[int(path[t + 1]), int(t + 1)]
        return path, delta, phi

    def score(self, X, y, sample_weight=None):
        pass
