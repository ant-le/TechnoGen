import pathlib

from torch.utils.data import Dataset
import torch.nn as nn
from BeatNet.BeatNet import BeatNet
from torch import Tensor
import torchaudio
import torch
import numpy as np
import librosa


class TechnoGenDataset(Dataset):
    def __init__(
        self,
        audio_dir: pathlib.PosixPath,
        num_samples: int,
        transformation: nn.Module,
        sapmple_rate: int,
    ):
        self.audio_files = {
            idx: audio_file
            for idx, audio_file in enumerate(list(audio_dir.glob("*.wav")))
        }
        self.transformation = transformation
        self.num_samples = num_samples
        self.sapmple_rate = sapmple_rate

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, index):
        if index >= len(self.audio_files):
            raise IndexError(
                "Index is out of range! Chosen index exceeds number of audio files."
            )
        audio_samle_path = self.audio_files[index]
        signal, sr = torchaudio.load(audio_samle_path)
        signal = self._resample(signal, self.sample_rate)
        # signal = self._mix_down(signal)
        # beats = self.get_beat_locations(audio_samle_path)
        # signal_list = self._split_by_beats(beats)
        #  signal = self._right_pad(signal)
        #  signal = self.transformation(signal)
        return signal

    def get_beat_locations(self, audio_samle_path) -> [float]:
        # probably I will need my own algorithm since I would
        # need the same sampling rate for it to work properly
        result = self.beat_estimator.process(str(audio_samle_path))
        return result[:, 0]

    def _resample(self, signal: Tensor, sr: int) -> Tensor:
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down(self, signal: Tensor) -> Tensor:
        # aggregates different audio channels
        if signal.shape[0] > 1:
            # (n_channels, n_samples) -> (1, n_samples)
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _split_by_beats(signal: Tensor, beat_locations) -> [Tensor]:
        seq = []
        old_beat = [0]
        beat_locations.extend([len(signal)])
        for beat in beat_locations[1:]:
            seq.append(signal[:, old_beat.type(torch.int64) : beat.type(torch.int64)])
            old_beat = beat
        return seq


if __name__ == "__main__":
    NUM_SAMPLES = 8
    SAMPLE_RATE = 44600
    PATH = pathlib.Path.home() / "Documents" / "Music" / "Samoh"
    NUM_SAMPLES = 1_200_000
    TRANSORMATION = nn.Identity()
    BEAT_ESTIMATOR = BeatNet(
        1, mode="offline", inference_model="DBN", plot=[], thread=False
    )

    data = TechnoGenDataset(
        PATH, NUM_SAMPLES, TRANSORMATION, SAMPLE_RATE, BEAT_ESTIMATOR
    )
    sample = data[0]
