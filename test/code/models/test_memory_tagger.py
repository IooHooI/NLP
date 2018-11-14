import unittest

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd

from source.code.utils.preprocessing import filtrations
from source.code.utils.preprocessing import iob3bio

from source.code.transformers.sentenceextractor import SentenceExtractor

from source.code.models.memorytagger import MemoryTagger

from sklearn.model_selection import train_test_split

from seqeval.metrics import classification_report as seqeval_classification_report


class TestMemoryTagger(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        data.ner_tag = iob3bio(data.ner_tag.values)

        cls.features = [
            'token',
            'pos_tag',
            'lemma'
        ]

        X, y = SentenceExtractor(features=cls.features, target='ner_tag').fit_transform(data)

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    def test_case_1(self):
        tagger = MemoryTagger()

        tagger.fit(self.X_train, self.y_train)

        self.X_test = [sentence for sentence in self.X_test if len(sentence) > 0]

        self.y_test = [sentence.tolist() for sentence in self.y_test if len(sentence) > 0]

        y_pred = tagger.predict(self.X_test)

        seqeval_report = seqeval_classification_report(y_pred=y_pred, y_true=self.y_test)

        print(seqeval_report)


if __name__ == '__main__':
    unittest.main()
