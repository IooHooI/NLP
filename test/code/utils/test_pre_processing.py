import unittest

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts_as_pd
from source.code.utils.preprocessing import filtrations
from source.code.utils.preprocessing import additional_features


class TestPreProcessingMethods(unittest.TestCase):

    def test_case_1(self):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        self.assertTrue(9167 == len(folders))

    def test_case_2(self):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')

        self.assertTrue(1231279 == len(data))

    def test_case_3(self):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')
        data = filtrations(data, with_dots=True)

        self.assertTrue(780339 == len(data))

    def test_case_4(self):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        data = get_tagged_texts_as_pd(folders, '../../../data/datasets/gmb-2.2.0')
        data = filtrations(data, with_dots=True)
        data = additional_features(data)

        self.assertTrue(780339 == len(data))


if __name__ == '__main__':
    unittest.main()
