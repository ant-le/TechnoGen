import unittest
import torch
import sys

sys.path.append(".")

from dataset.data_generator import create_sequences


class TestDataset(unittest.TestCase):
    def test_beat_splitting(self):
        """Test if spliting by beats does
        not have inherent mistakes"""


if __name__ == "__main__":
    unittest.main()
