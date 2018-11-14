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

    @classmethod
    def setUpClass(cls):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        data.ner_tag = iob3bio(data.ner_tag.values)

        data = additional_features(df=data)

        cls.features = data.columns.values.tolist()

        cls.features.remove('ner_tag')

        cls.features.remove('word_net_sense_number')

        cls.features.remove('verb_net_roles')

        cls.features.remove('semantic_relation')

        cls.features.remove('animacy_tag')

        cls.features.remove('super_tag')

        cls.features.remove('lambda_dsr')

        X, y = SentenceExtractor(features=cls.features, target='ner_tag').fit_transform(data)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        cls.X_train = [sentence for sentence in cls.X_train if len(sentence) > 0]

        cls.y_train = [sentence.tolist() for sentence in cls.y_train if len(sentence) > 0]

        cls.X_test = [sentence for sentence in cls.X_test if len(sentence) > 0]

        cls.y_test = [sentence.tolist() for sentence in cls.y_test if len(sentence) > 0]

    def test_case_1_with_1_feature(self):
        X_train_1_f = [sentence[:, self.features.index('lemma')] for sentence in self.X_train]

        X_test_1_f = [sentence[:, self.features.index('lemma')] for sentence in self.X_test]

        pipeline = Pipeline([
            ('transform', CRFTransformer(features=[
                'lemma'
            ])),
            ('fit', CRFTagger())
        ])

        pipeline.fit(X_train_1_f, self.y_train)

        y_pred = pipeline.predict(X_test_1_f)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)

    def test_case_2_with_2_features(self):
        X_train_2_fs = [
            sentence[:, self.features.index('token'):self.features.index('pos_tag') + 1] for sentence in self.X_train
        ]

        X_test_2_fs = [
            sentence[:, self.features.index('token'):self.features.index('pos_tag') + 1] for sentence in self.X_test
        ]

        pipeline = Pipeline([
            ('transform', CRFTransformer(features=[
                'token',
                'pos_tag'
            ])),
            ('fit', CRFTagger())
        ])

        pipeline.fit(X_train_2_fs, self.y_train)

        y_pred = pipeline.predict(X_test_2_fs)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)

    def test_case_3_with_3_additional_features(self):
        X_train_3_fs = [
            sentence[:, self.features.index('token'):self.features.index('lemma') + 1] for sentence in self.X_train
        ]

        X_test_3_fs = [
            sentence[:, self.features.index('token'):self.features.index('lemma') + 1] for sentence in self.X_test
        ]

        pipeline = Pipeline([
            ('transform', CRFTransformer(features=[
                'token',
                'pos_tag',
                'lemma'
            ])),
            ('fit', CRFTagger())
        ])

        pipeline.fit(X_train_3_fs, self.y_train)

        y_pred = pipeline.predict(X_test_3_fs)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)

    def test_case_4_with_4_additional_features(self):
        X_train_4_fs = [
            sentence[:, self.features.index('token'):self.features.index('is_title') + 1] for sentence in self.X_train
        ]

        X_test_4_fs = [
            sentence[:, self.features.index('token'):self.features.index('is_title') + 1] for sentence in self.X_test
        ]

        pipeline = Pipeline([
            ('transform', CRFTransformer(features=[
                'token',
                'pos_tag',
                'lemma',
                'is_title'
            ])),
            ('fit', CRFTagger())
        ])

        pipeline.fit(X_train_4_fs, self.y_train)

        y_pred = pipeline.predict(X_test_4_fs)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)

    def test_case_5_with_5_additional_features(self):
        X_train_5_fs = [
            sentence[:, self.features.index('token'):self.features.index('contains_digits') + 1] for sentence in self.X_train
        ]

        X_test_5_fs = [
            sentence[:, self.features.index('token'):self.features.index('contains_digits') + 1] for sentence in self.X_test
        ]

        pipeline = Pipeline([
            ('transform', CRFTransformer(features=[
                'token',
                'pos_tag',
                'lemma',
                'is_title',
                'contains_digits'
            ])),
            ('fit', CRFTagger())
        ])

        pipeline.fit(X_train_5_fs, self.y_train)

        y_pred = pipeline.predict(X_test_5_fs)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)

    def test_case_6_with_6_additional_features(self):
        X_train_6_fs = [
            sentence[:, self.features.index('token'):self.features.index('word_len') + 1] for sentence in self.X_train
        ]

        X_test_6_fs = [
            sentence[:, self.features.index('token'):self.features.index('word_len') + 1] for sentence in self.X_test
        ]

        pipeline = Pipeline([
            ('transform', CRFTransformer(features=[
                'token',
                'pos_tag',
                'lemma',
                'is_title',
                'contains_digits',
                'word_len'
            ])),
            ('fit', CRFTagger())
        ])

        pipeline.fit(X_train_6_fs, self.y_train)

        y_pred = pipeline.predict(X_test_6_fs)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)


if __name__ == '__main__':
    unittest.main()
