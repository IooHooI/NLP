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

        X, y = SentenceExtractor(features=[
            'token',
            'pos_tag',
            'lemma'
        ]).fit_transform(data)

        lemma_sentence_lenghts = list(map(len, X))

        tag_sentence_lenghts = list(map(len, y))

        self.assertTrue(
            all(len_lemmas == len_tags for len_lemmas, len_tags in zip(lemma_sentence_lenghts, tag_sentence_lenghts))
        )


if __name__ == '__main__':
    unittest.main()
