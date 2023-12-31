import h5py, torch
from torch.utils.data import Dataset

from pathlib import PosixPath

from dataset.data_generator import generate_dataset_file


class TechnoGenDataset(Dataset):
    """Dataset Class nadling all audio files used to train
    and evaluate the model. It operates on a HDF5 file format
    and generates lookup table for each song (splits) stored
    as an array. If a file with the desired data specifications
    is not available yet, it will be generated automatically.
    """

    def __init__(self, config):
        super(TechnoGenDataset, self).__init__()

        self.audio_lookup = {}
        sample_rate = config["sample_rate"]
        hop_size = config["hop_size"]
        self.num_samples = sample_rate * hop_size
        self.data_path = PosixPath(
            "dataset", "data", f"techno_{self.num_samples}_{sample_rate}.h5"
        )
        # create a lookup table for quickly acessing samples
        if not self.data_path.exists():
            print(
                f"--- Dataset does not exist for sample rate {sample_rate} and hop size {hop_size}. Creating the dataset now..."
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
            print("--- Something went wrong when reading the data file!")

    def __len__(self) -> int:
        return list(self.audio_lookup.values())[-1]

    def __getitem__(self, index) -> torch.Tensor:
        curr_count = 0

        # find key corresponding to index in lookup table
        for key, count in self.audio_lookup.items():
            if index < count:
                break
            curr_count = count
        lookup_idx = index - curr_count

        # return the numpy array at index
        with h5py.File(self.data_path, "r") as f:
            audio_wave = f[self.num_samples][key][()]
            signal = torch.from_numpy(audio_wave[lookup_idx, :])
        return signal
