import unittest

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd

from source.code.utils.preprocessing import additional_features
from source.code.utils.preprocessing import filtrations
from source.code.utils.preprocessing import iob3bio

from source.code.transformers.sentenceextractor import SentenceExtractor
from source.code.transformers.crftransformer import CRFTransformer

from source.code.models.crftagger import CRFTagger

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from seqeval.metrics import classification_report as seqeval_classification_report


class TestSRFTransformer(unittest.TestCase):

    def _test_step(self, from_, to_):
        X_train_f = [
            sentence[:, self.features.index(from_):self.features.index(to_) + 1] for sentence in self.X_train
        ]

        X_test_f = [
            sentence[:, self.features.index(from_):self.features.index(to_) + 1] for sentence in self.X_test
        ]

        pipeline = Pipeline([
            ('transform', CRFTransformer(
                features=self.features[self.features.index(from_):self.features.index(to_) + 1]
            )),
            ('fit', CRFTagger())
        ])

        pipeline.fit(X_train_f, self.y_train)

        y_pred = pipeline.predict(X_test_f)

        return y_pred

    def _test_and_validation_step(self, from_, to_):
        y_pred = self._test_step(from_, to_)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)

    @classmethod
    def setUpClass(cls):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        data.ner_tag = iob3bio(data.ner_tag.values)

        data = additional_features(df=data)

        # features list:
        cls.features = [
            'token',
            'lemma',
            'pos_tag',
            'is_title',
            'contains_digits',
            'word_len',
            'suffix',
            'prefix',
            'prev_pos_tag',
            'prev_is_title',
            'prev_contains_digits',
            'prev_word_len',
            'prev_suffix',
            'prev_prefix',
            'next_pos_tag',
            'next_is_title',
            'next_contains_digits',
            'next_word_len',
            'next_suffix',
            'next_prefix'
        ]

        X, y = SentenceExtractor(features=cls.features, target='ner_tag').fit_transform(data)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        cls.X_train = [sentence for sentence in cls.X_train if len(sentence) > 0]

        cls.y_train = [sentence.tolist() for sentence in cls.y_train if len(sentence) > 0]

        cls.X_test = [sentence for sentence in cls.X_test if len(sentence) > 0]

        cls.y_test = [sentence.tolist() for sentence in cls.y_test if len(sentence) > 0]

    def test_case_1_with_1_feature(self):
        self._test_and_validation_step('lemma', 'lemma')

    def test_case_2_with_2_features(self):
        self._test_and_validation_step('pos_tag', 'is_title')

    def test_case_3_with_3_additional_features(self):
        self._test_and_validation_step('pos_tag', 'contains_digits')

    def test_case_4_with_4_additional_features(self):
        self._test_and_validation_step('pos_tag', 'word_len')

    def test_case_5_with_5_additional_features(self):
        self._test_and_validation_step('pos_tag', 'suffix')

    def test_case_6_with_6_additional_features(self):
        self._test_and_validation_step('pos_tag', 'prefix')

    def test_case_7_with_7_additional_features(self):
        self._test_and_validation_step('pos_tag', 'prev_pos_tag')

    def test_case_8_with_8_additional_features(self):
        self._test_and_validation_step('pos_tag', 'prev_is_title')

    def test_case_9_with_9_additional_features(self):
        self._test_and_validation_step('pos_tag', 'prev_contains_digits')

    def test_case_10_with_10_additional_features(self):
        self._test_and_validation_step('pos_tag', 'prev_word_len')

    def test_case_11_with_11_additional_features(self):
        self._test_and_validation_step('pos_tag', 'prev_suffix')

    def test_case_12_with_12_additional_features(self):
        self._test_and_validation_step('pos_tag', 'prev_prefix')

    def test_case_13_with_13_additional_features(self):
        self._test_and_validation_step('pos_tag', 'next_pos_tag')

    def test_case_14_with_14_additional_features(self):
        self._test_and_validation_step('pos_tag', 'next_is_title')

    def test_case_15_with_15_additional_features(self):
        self._test_and_validation_step('pos_tag', 'next_contains_digits')

    def test_case_16_with_16_additional_features(self):
        self._test_and_validation_step('pos_tag', 'next_word_len')

    def test_case_17_with_17_additional_features(self):
        self._test_and_validation_step('pos_tag', 'next_suffix')

    def test_case_18_with_18_additional_features(self):
        self._test_and_validation_step('pos_tag', 'next_prefix')


if __name__ == '__main__':
    unittest.main()
