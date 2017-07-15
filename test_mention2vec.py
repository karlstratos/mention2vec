import unittest
from mention2vec import get_boundaries
from mention2vec import label_bio

class TestBoundaryExtraction(unittest.TestCase):
    """Test the correctness of boundary extraction from BIO sequences."""

    def test_raw_bio_proper(self):
        """Works on properly formatted raw sequences?"""

        multiple_Bs = ['O', 'B', 'B', 'I', 'B', 'O']
        front_back  = ['B', 'I', 'I', 'O', 'O', 'B']

        self.assertEqual(get_boundaries(multiple_Bs),
                         [(1, 1, None), (2, 3, None), (4, 4, None)])
        self.assertEqual(get_boundaries(front_back),
                         [(0, 2, None), (5, 5, None)])

    def test_raw_bio_improper(self):
        """Works on improperly formatted raw sequences (as intended)?"""

        start_I  = ['O', 'I', 'I', 'B', 'I', 'O', 'I', 'I']
        all_I    = ['I', 'I', 'I', 'I', 'I', 'I', 'I', 'I']
        scramble = ['I', 'I', 'B', 'I', 'O', 'B', 'B', 'O']

        self.assertEqual(get_boundaries(start_I),
                         [(1, 2, None), (3, 4, None), (6, 7, None)])
        self.assertEqual(get_boundaries(all_I), [(0, 7, None)])
        self.assertEqual(get_boundaries(scramble),
                         [(0, 1, None), (2, 3, None), (5, 5, None),
                          (6, 6, None)])

    def test_labeled_bio_proper(self):
        """Works on properly formatted labeled sequences?"""

        multiple_Bs_same = ['O', 'B-ORG', 'B-ORG', 'I-ORG', 'B-ORG', 'O']
        multiple_Bs_diff = ['O', 'B-PER', 'B-LOC', 'I-LOC', 'B-MISC', 'O']
        front_back       = ['B-PER', 'I-PER', 'I-PER', 'O', 'O', 'B-LOC']

        self.assertEqual(get_boundaries(multiple_Bs_same),
                         [(1, 1, 'ORG'), (2, 3, 'ORG'), (4, 4, 'ORG')])
        self.assertEqual(get_boundaries(multiple_Bs_diff),
                         [(1, 1, 'PER'), (2, 3, 'LOC'), (4, 4, 'MISC')])
        self.assertEqual(get_boundaries(front_back),
                         [(0, 2, 'PER'), (5, 5, 'LOC')])

    def test_labeled_bio_improper(self):
        """Works on improperly formatted labeled sequences (as intended)?"""

        wrong_BIO \
            = ['O', 'I-ORG', 'B-ORG', 'O', 'I-ORG', 'B-ORG', 'B-ORG', 'I-ORG']
        wrong_entity \
            = ['O', 'B-ORG', 'I-PER', 'O', 'B-ORG', 'B-LOC', 'I-ORG', 'I-LOC']
        wrong_BIO_entity \
            = ['I-ORG', 'I-PER', 'I-LOC', 'B-LOC', 'I-LOC', 'I-ORG', 'I-LOC']


        self.assertEqual(get_boundaries(wrong_BIO),
                         [(1, 1, 'ORG'), (2, 2, 'ORG'), (4, 4, 'ORG'),
                          (5, 5, 'ORG'), (6, 7, 'ORG')])
        self.assertEqual(get_boundaries(wrong_entity),
                         [(1, 1, 'ORG'), (2, 2, 'PER'), (4, 4, 'ORG'),
                          (5, 5, 'LOC'), (6, 6, 'ORG'), (7, 7, 'LOC')])
        self.assertEqual(get_boundaries(wrong_BIO_entity),
                         [(0, 0, 'ORG'), (1, 1, 'PER'), (2, 2, 'LOC'),
                          (3, 4, 'LOC'), (5, 5, 'ORG'), (6, 6, 'LOC')])

class TestBIOLabeling(unittest.TestCase):
    """Test the correctness of BIO labeling."""

    def test_proper(self):
        """Proper example"""
        bio  = ['O', 'B', 'B', 'I', 'B', 'O']
        ents = ['PER', 'ORG', 'PER']

        self.assertEqual(label_bio(bio, ents),
                         ['O', 'B-PER', 'B-ORG', 'I-ORG', 'B-PER', 'O'])

    def test_improper1(self):
        """Improper example 1"""
        bio  = ['I', 'I', 'I', 'I', 'I', 'I']
        ents = ['PER']

        self.assertEqual(label_bio(bio, ents),
                         ['I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER', 'I-PER'])

    def test_improper2(self):
        """Improper example 2"""
        bio  = ['I', 'B', 'I', 'O', 'I', 'B', 'I']
        ents = ['PER', 'PER', 'LOC', 'PER']

        self.assertEqual(label_bio(bio, ents),
                         ['I-PER', 'B-PER', 'I-PER', 'O', 'I-LOC', 'B-PER',
                          'I-PER'])

if __name__ == '__main__':
    unittest.main()
