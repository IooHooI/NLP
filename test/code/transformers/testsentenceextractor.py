import unittest

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd
from source.code.utils.preprocessing import filtrations
from source.code.transformers.sentenceextractor import SentenceExtractor


class TestSentenceExtractor(unittest.TestCase):

    def test_case_1(self):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')
        data = filtrations(data, with_dots=True)

        X, y = SentenceExtractor().fit_transform(data)

        self.assertEqual(len(X), len(y))


if __name__ == '__main__':
    unittest.main()
