import h5py, torchaudio, torch
from torch import Tensor

from tqdm import tqdm
from pathlib import PosixPath, Path


def resample(signal: Tensor, from_sr: int, to_sr: int, debug=False) -> Tensor:
    """Resamples all audio files to a given sampling rate defined
    in the configuration file"""
    if from_sr != to_sr:
        resampler = torchaudio.transforms.Resample(from_sr, to_sr)
        signal = resampler(signal)
        if debug:
            tqdm.write(f"Track was resampled from original sr={from_sr}")
    return signal


def mix_down(signal: Tensor) -> Tensor:
    """Mixes down stereo signals to audio if necessary"""
    if signal.shape[0] > 1:
        # (n_channels, n_samples) -> (1, n_samples)
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def split_by_time(signal: Tensor, k_seconds: int, same_rate: int):
    num_samples = k_seconds * same_rate

    splits = torch.split(signal, num_samples, dim=1)

    # drop ending of the song due to different size
    splits = torch.stack(list(splits[:-1]), dim=0)

    # (split, 1, num_samples) -> (split, num_samples)
    splits.reshape(splits.shape[0], splits.shape[2])

    return splits


def apply_padding(signal: Tensor, num_samples: int, idx: int) -> Tensor:
    """Functions that brings all signals into the same format (lenght). It
    cuts longer files after the maximum lenght defined in num_samples and
    applies padding to shorter files. Warinings are raised when cutting
    since such behavior is not desirable."""

    n_missing_samples = num_samples - signal.shape[1]
    if n_missing_samples < 0:
        # cut sample with warning
        signal = signal[:, :num_samples]
        tqdm.write(
            f"Warning: Track {idx+1} had {abs(n_missing_samples)} too many samples and was cut!"
        )

    if n_missing_samples > 0:
        # rigth padding
        dims = (0, n_missing_samples)
        signal = torch.nn.functional.pad(signal, dims)
    return signal


def generate_dataset_file(config):
    required_keys = ["sample_rate", "hop_size", "channels"]

    if not all(key in config for key in required_keys):
        raise ValueError(
            f"Config file must contain properties {','.join(required_keys)}."
        )
    print("Configuration file loaded successfully! The chosen specifications are:")

    ######## Compute resulting sizes #######
    for key in required_keys:
        print("----> {}: {}".format(key, config[key]))

    ######## generate data #######

    # Define further configurations and initiate feature dict for storing restults
    path = (
        PosixPath(config["offline_data_path"])
        if PosixPath(config["offline_data_path"]).exists()
        else config["online_data_path"]
    )

    # get paths of all audio files in directory
    song_paths = [str(song_file) for song_file in list(path.rglob("*.wav"))]
    song_paths.extend([str(song_file) for song_file in list(path.rglob("*.mp3"))])
    assert len(song_paths) > 0

    sample_rate = config["sample_rate"]
    # Process and store each song
    features = {}
    for idx, song_path in enumerate(
        tqdm(
            (song_paths),
            desc="Downloading and Processing tracks",
            leave=True,
            position=0,
        )
    ):
        # Loading and preparing track (audio file)
        audio_wave, sr = torchaudio.load(song_path)
        audio_wave = resample(
            audio_wave, from_sr=sr, to_sr=sample_rate, debug=config["verbose"]
        )
        audio_wave = mix_down(audio_wave)

        audio_seq = split_by_time(
            audio_wave,
            k_seconds=config["hop_size"],
            same_rate=sample_rate,
        )

        features[str(idx)] = audio_seq

    # store data in hdf5 format in data directory
    data_path = Path(__file__).parent / "data"
    data_path.mkdir(exist_ok=True)

    num_samples = sample_rate * config["hop_size"]
    with h5py.File(
        data_path / f"techno_{num_samples}_{sample_rate}.h5",
        "w",
    ) as f:
        grp = f.create_group(f"{num_samples}")
        for idx, track in features.items():
            grp.create_dataset(idx, data=track.numpy())


if __name__ == "__main__":
    pass
