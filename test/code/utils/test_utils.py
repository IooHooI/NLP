import unittest

from source.code.utils.utils import filter_by_subcorpus
from source.code.utils.utils import get_tagged_texts


class TestUtils(unittest.TestCase):

    def test_case_1(self):
        folders = filter_by_subcorpus('../../../data/datasets/gmb-2.2.0', 'subcorpus: Voice of America')

        texts = get_tagged_texts(folders, '../../../data/datasets/gmb-2.2.0')

        self.assertTrue(9167, len(folders))


if __name__ == '__main__':
    unittest.main()
