import unittest

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd

from source.code.utils.preprocessing import filtrations
from source.code.utils.preprocessing import iob3bio

from source.code.transformers.sentenceextractor import SentenceExtractor
from source.code.transformers.crftransformer import CRFTransformer


class TestSRFTransformer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        data.ner_tag = iob3bio(data.ner_tag.values)

        cls.X, cls.y = SentenceExtractor(
            features=[
                'token',
                'pos_tag',
                'lemma'
            ],
            target='ner_tag'
        ).fit_transform(data)

    def test_case_1(self):
        X = CRFTransformer(features=[
            'token',
            'pos_tag',
            'lemma'
        ]).fit_transform(self.X, self.y)

        self.assertEqual(len(self.X), len(X))


if __name__ == '__main__':
    unittest.main()
