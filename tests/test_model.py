import unittest
import torch
import sys

sys.path.append(".")
from model.vqvae.vqvae import VQVAE


class TestModel(unittest.TestCase):
    def test_foward_run(self):
        seconds = 8
        audio_wave = torch.randn(32, 1, 44_000 * seconds)

        model = VQVAE((1, 44_000 * seconds))
        output_wave, loss, _ = model.forward(audio_wave)
        self.assertEqual(audio_wave.shape, output_wave.shape)
        self.assertEqual(loss.shape, torch.Size([]))


if __name__ == "__main__":
    unittest.main()
