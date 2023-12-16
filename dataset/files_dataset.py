import h5py
import torch
from torch.utils.data import Dataset
import torch

from data_generator import generate_dataset_file
from pathlib import PosixPath


# maybe move to handle all data generation
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

    def __init__(self, config):
        super(TechnoGenDataset, self).__init__()

        self.audio_lookup = {}
        self.data_path = PosixPath(
            "dataset", "data", f"techno_{config['num_samples']}.h5"
        )
        # create a lookup table for quickly acessing samples
        if not self.data_path.exists():
            print(
                f"Dataset does not exist for sample rate {config['sample_rate']} and hop size {config['hop_size']}. Creating the dataset now..."
            )
            generate_dataset_file(config)

        try:
            with h5py.File(self.data_path, "r") as f:  #
                self.num_samples = list(f.keys())[0]
                for idx in list(f[self.num_samples]):
                    self.audio_lookup[idx] = int(f[self.num_samples][idx].shape[0])
                    if int(idx) != 0:
                        self.audio_lookup[idx] += list(self.audio_lookup.values())[-2]
        except Exception as e:
            print("Something went wrong when reading the data file!")

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
            audio_wave = f[self.num_samples][key][()]
            signal = torch.from_numpy(audio_wave[lookup_idx, :])
            signal = signal.reshape(1, signal.shape[0])  # (1, num_samples)

        return signal
