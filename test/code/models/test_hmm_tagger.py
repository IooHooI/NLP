import unittest

from sklearn.model_selection import train_test_split

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd

from source.code.utils.preprocessing import filtrations
from source.code.utils.preprocessing import iob3bio

from source.code.transformers.sentenceextractor import SentenceExtractor

from source.code.models.hmmtagger import HMMTagger


class TestHMMTagger(unittest.TestCase):

    def test_case_1(self):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        data.ner_tag = iob3bio(data.ner_tag.values)

        hmm_tagger = HMMTagger()

        X, y = SentenceExtractor(features=[
            'token',
            'pos_tag',
            'lemma'
        ]).fit_transform(data)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        hmm_tagger.fit(X_train, y_train)




if __name__ == '__main__':
    unittest.main()
