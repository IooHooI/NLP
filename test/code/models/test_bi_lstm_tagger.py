import unittest

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd

from source.code.utils.preprocessing import additional_features
from source.code.utils.preprocessing import filtrations
from source.code.utils.preprocessing import iob3bio

from source.code.transformers.sentenceextractor import SentenceExtractor

from source.code.models.bilstmtagger import BiLSTMTagger

from sklearn.model_selection import train_test_split

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

        estimator = BiLSTMTagger(checkpoint_dir='../../../data/datasets/keras_model/')

        estimator.fit(X_train_1_f, self.y_train)

        y_pred = estimator.predict(X_test_1_f)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)


if __name__ == '__main__':
    unittest.main()
