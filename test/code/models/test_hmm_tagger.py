import unittest

from sklearn.model_selection import train_test_split

from seqeval.metrics import classification_report as seqeval_classification_report

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd

from source.code.utils.preprocessing import filtrations
from source.code.utils.preprocessing import iob3bio

from source.code.transformers.sentenceextractor import SentenceExtractor

from source.code.models.hmmtagger import HMMTagger


class TestHMMTagger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        data.ner_tag = iob3bio(data.ner_tag.values)

        X, y = SentenceExtractor(
            features=[
                'token',
                'pos_tag',
                'lemma'
            ],
            target='ner_tag'
        ).fit_transform(data)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def test_case_1_without_additional_features(self):
        X_train_1_f = [sentence[:, 0] for sentence in self.X_train]

        X_test_1_f = [sentence[:, 0] for sentence in self.X_test]

        hmm_tagger = HMMTagger()

        hmm_tagger.fit(X_train_1_f, self.y_train)

        X_test_1_f = [sentence for sentence in X_test_1_f if len(sentence) > 0]

        self.y_test = [sentence.tolist() for sentence in self.y_test if len(sentence) > 0]

        self.y_pred = hmm_tagger.predict(X_test_1_f)

        seqeval_report = seqeval_classification_report(y_pred=self.y_pred, y_true=self.y_test)

        print(seqeval_report)

    def test_case_2_with_1_additional_feature(self):
        X_train_2_fs = [sentence[:, 0:2] for sentence in self.X_train]

        X_test_2_fs = [sentence[:, 0:2] for sentence in self.X_test]

        hmm_tagger = HMMTagger(features=[
            'token',
            'pos_tag'
        ])

        hmm_tagger.fit(X_train_2_fs, self.y_train)

        X_test_2_fs = [sentence for sentence in X_test_2_fs if len(sentence) > 0]

        self.y_test = [sentence.tolist() for sentence in self.y_test if len(sentence) > 0]

        self.y_pred = hmm_tagger.predict(X_test_2_fs)

        seqeval_report = seqeval_classification_report(y_pred=self.y_pred, y_true=self.y_test)

        print(seqeval_report)

    def test_case_2_with_2_additional_features(self):
        hmm_tagger = HMMTagger(features=[
            'token',
            'pos_tag',
            'lemma'
        ])

        hmm_tagger.fit(self.X_train, self.y_train)

        self.X_test = [sentence for sentence in self.X_test if len(sentence) > 0]

        self.y_test = [sentence.tolist() for sentence in self.y_test if len(sentence) > 0]

        self.y_pred = hmm_tagger.predict(self.X_test)

        seqeval_report = seqeval_classification_report(y_pred=self.y_pred, y_true=self.y_test)

        print(seqeval_report)


if __name__ == '__main__':
    unittest.main()
