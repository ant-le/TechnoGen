import unittest
import torch
import sys

sys.path.append(".")
from model.vqvae.vqvae import VQVAE


class TestModel(unittest.TestCase):
    def test_foward_run(self):
        audio_wave = torch.randn(25, 1, 44_100 * 8)

        model = VQVAE((1, 44_100 * 9))
        output_wave = model.forward(audio_wave)
        self.assertEqual(audio_wave.shape, output_wave.shape)


if __name__ == "__main__":
    unittest.main()
