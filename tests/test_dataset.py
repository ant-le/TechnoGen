import unittest
import torch
import sys

sys.path.append(".")

from dataset.data_processor import DataLoader


class TestDataset(unittest.TestCase):
    """Unittests for all operations in dataset folder."""

    def test_generator(self):
        pass

    def test_processor(self):
        trai_loader = DataLoader()

    def test_files_dataset(self):
        # processor depends on dataset so dataset will be testes seperately
        pass


if __name__ == "__main__":
    unittest.main()
