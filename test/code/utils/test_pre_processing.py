import unittest

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd
from source.code.utils.preprocessing import filtrations
from source.code.utils.preprocessing import additional_features
from source.code.utils.preprocessing import iob3bio


class TestPreProcessingMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

    def test_case_1(self):
        self.assertTrue(9167 == len(self.folders))

    def test_case_2(self):
        data = get_tagged_texts_as_pd(self.folders, '../../../data/datasets/gmb-2.2.0')

        self.assertTrue(1231279 == len(data))

    def test_case_3(self):
        data = get_tagged_texts_as_pd(self.folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        self.assertTrue(780339 == len(data))

    def test_case_4(self):
        data = get_tagged_texts_as_pd(self.folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        data = additional_features(data)

        self.assertTrue(780339 == len(data))

    def test_case_5(self):
        data = get_tagged_texts_as_pd(self.folders, '../../../data/datasets/gmb-2.2.0')

        data = filtrations(data, with_dots=True)

        bio_ner_tags = iob3bio(data.ner_tag.values.tolist())

        self.assertTrue(len(bio_ner_tags) == len(data))

    def test_case_6(self):
        ner_tags = iob3bio([
            'per-tit',
            'per-nam',
            'per-nam',
            'per-nam',
            'O',
            'O',
            'O',
            'O',
            'gpe-tit',
            'gpe-nam',
            'gpe-nam',
            'gpe-tit',
            'gpe-tit',
            'O',
            'O',
            'O',
            'O',
            'O',
            'O'
        ])
        self.assertEqual(
            ner_tags,
            [
                'B-per',
                'B-per',
                'I-per',
                'I-per',
                'O',
                'O',
                'O',
                'O',
                'B-gpe',
                'B-gpe',
                'I-gpe',
                'B-gpe',
                'I-gpe',
                'O',
                'O',
                'O',
                'O',
                'O',
                'O'
            ],
            'Something is wrong!!!'
        )


if __name__ == '__main__':
    unittest.main()
