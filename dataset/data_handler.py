import torch, torchaudio, sys

sys.path.append(".")

from pathlib import Path
from dataset.utils import resample, mix_down
from torch.utils.data import Dataset


class TechnoGenDataset(Dataset):
    """Dataset Class handling all audio files used to train
    and evaluate the model. It operates on a HDF5 file format
    and generates lookup table for each song (splits) stored
    as an array. If a file with the desired data specifications
    is not available yet, it will be generated automatically.
    """

    def __init__(self, config):
        super(TechnoGenDataset, self).__init__()

        self.hop_size = config["hop_size"]
        self.channels = config["channels"]
        self.n_samples = config["n_samples"]
        self.sample_rate = config["sample_rate"]
        self._create_dataset(config)

    def __len__(self) -> int:
        return len(self.song_paths) * self.n_samples

    def __getitem__(self, index) -> torch.Tensor:
        audio_wave = self._getTrack(index)
        return self._getSample(index, audio_wave)

    def _getTrack(self, index):
        track_number = index // self.n_samples
        signal, sr = torchaudio.load(
            self.song_paths[track_number]
        )  # TODO: don't load full song?
        signal = resample(signal, from_sr=sr, to_sr=self.sample_rate)
        signal = mix_down(signal)
        return signal

    def _getSample(self, index, signal):
        split = signal.shape[1] // self.sample_rate // self.n_samples
        idx = (split * (index % self.n_samples) + split // 3) * self.sample_rate
        return signal[:, idx : idx + self.sample_rate * self.hop_size]

    def _create_dataset(self, config):
        self.song_paths = []
        allowed_file_ext = ["wav"]
        if config["all_filetypes"]:
            allowed_file_ext.extend(["mp3", "flac", "mp4", "aac"])

        for ext in allowed_file_ext:
            self.song_paths.extend(
                [
                    str(song_file)
                    for song_file in list(Path(config["path"]).rglob(f"*.{ext}"))
                ]
            )
        assert len(self.song_paths) > 0

        if config["limit"] is not None:
            print(config["limit"])
            self.song_paths = self.song_paths[: config["limit"]]
