from pathlib import Path
import h5py

from torch.utils.data import Dataset
import torch.nn as nn
import torch
import torchaudio


class TechnoGenDataset(Dataset):
    """Dataset Class for the containing all relevant
    tracks to train and evaluate the model. It accesses
    the .h5 - file with a generated lookup table that
    contains as a key an index value for each stored
    track and as a value a continuing count of how
    many sequences are stored in the the file in order
    to facilitate the needed implementation of the
    __getitem__ and len methods.
    """

    def __init__(
        self,
        transformation,
        sample_rate: int,
        k_beats: int,
        train: bool = True,
        limit: int = None,
    ):
        super().__init__()
        self.limit = limit
        self.train = train
        self.transformation = transformation
        self.sample_rate = sample_rate
        self.k_beats = k_beats

        # get path of dataset
        self.data_path = (
            Path(__file__).parent
            / "data"
            / f"techno_{self.sample_rate}_{self.k_beats}.h5"
        )

        # create a lookup table for quickly acessing samples
        if self.data_path.exists():
            self.audio_lookup = {}
            with h5py.File(self.data_path, "r") as f:
                # get object name
                self.group_key = list(f.keys())[0]
                # get all indices of tracks
                track_idxs = list(f[self.group_key])

                # iterate through all tracks and store information about how many
                # sequences they contain
                for idx in track_idxs:
                    self.audio_lookup[idx] = int(f[self.group_key][idx].shape[0])
                    if int(idx) != 0:
                        self.audio_lookup[idx] += list(self.audio_lookup.values())[-2]

                        if self.limit:
                            if self.audio_lookup[idx] > self.limit:
                                self.audio_lookup[idx] = self.limit
                                break  # stop if limit is reached
        else:
            print(
                "Dataset was not found! You need to first generate it in 'dataset_generation.py' using your chosen sample and split rates!"
            )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

        return list(self.audio_lookup.values())[-1]

    def __getitem__(self, index) -> torch.Tensor:
        """Return the sample at index."""

        # find corresponding index in h5-file
        curr_count = 0
        for key, count in self.audio_lookup.items():
            if index < count:
                break
            curr_count = count
        lookup_idx = index - curr_count

        # read the numpy array at index
        with h5py.File(self.data_path, "r") as f:
            audio_wave = f[self.group_key][key][()]
            signal = torch.from_numpy(audio_wave[lookup_idx, :])
            signal = signal.reshape(1, signal.shape[0])  # (1, num_samples)

        # apply transformation of input
        signal = self.transformation(signal)
        return signal


if __name__ == "__main__":
    identity = nn.Identity()
    spectogram = torchaudio.transforms.Spectrogram()
    data = TechnoGenDataset(
        transformation=spectogram, sample_rate=44_000, k_beats=16, limit=200
    )
