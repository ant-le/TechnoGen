import h5py

import torchaudio
import torch
from torch import Tensor


from tqdm import tqdm
from librosa.beat import beat_track
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


def get_beat_locations(
    signal: Tensor, sr: int, verbose: bool = False
) -> (float, [int]):
    """Calls librosas beat tracking estimator to find beat locattions
    for a given signal. Returns estimated beats per minute (bpm) and
    a list with sample locations for beats.
    """

    tempo, beats = beat_track(
        y=signal.numpy().reshape(
            -1,
        ),
        sr=sr,
        start_bpm=140.0,
        units="samples",
    )
    if verbose:
        tqdm.write("Estimated Tempo of current track is {}".format(tempo))
    return tempo, beats


def split_by_time(signal: Tensor, k_seconds: int, same_rate: int):
    num_samples = k_seconds * same_rate

    splits = torch.split(signal, num_samples, dim=1)

    # drop ending of the song due to different size
    splits = torch.stack(list(splits[:-1]), dim=0)

    # (split, 1, num_samples) -> (split, num_samples)
    splits.reshape(splits.shape[0], splits.shape[2])

    return splits


def split_by_beats(
    signal: Tensor,
    beat_locations: [int],
    idx: int,
    k_beats: int,
    num_samples: int,
    verbose: bool = False,
) -> Tensor:
    """Splits a given audio signal (song) every k_beats beats. The
    frames before the first beat and after the last beat are hereby
    ommitted due to their large variation in signal lenght.
    """

    # TODO: move it to seperate test file where I don't need to pad
    seq_list = []
    padded_seq_list = []

    old_beat_idx = 0
    for beat_idx in range(1, len(beat_locations)):  # iterate over found beats
        if beat_idx % k_beats == 0:
            # find corresponding interval in signal
            sequence = signal[
                :, beat_locations[old_beat_idx] : beat_locations[beat_idx]
            ]
            seq_list.append(sequence)

            # apply padding to signal to get equal lenghs for all sequences
            # this is done here since we then don't need to loop over all track
            # sequences again later -> saves time and memory usage
            sequence = apply_padding(sequence, num_samples, idx)
            padded_seq_list.append(sequence)
            old_beat_idx = beat_idx

    # igonore all signals ...
    start = signal[:, : beat_locations[0]]  # ... before first beat
    ending = signal[:, beat_locations[old_beat_idx] :]  # ... after last beat

    # testing
    expected_count = (
        start.shape[1] + sum(map(lambda x: x.shape[1], seq_list)) + ending.shape[1]
    )
    assert expected_count == signal.shape[1]

    # Convert track back to Tensor (n_splits, num_samples)
    padded_track = torch.cat(padded_seq_list, dim=0)
    if verbose:
        tqdm.write(
            "Track successfully sequenced with shape: {0}".format(padded_track.shape)
        )

    return padded_track


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
    required_keys = ["min_tempo", "sample_rate", "hop_size", "channels"]

    if not all(key in config for key in required_keys):
        raise ValueError(
            f"Config file must contain properties {','.join(required_keys)}."
        )
    print("Configuration file loaded successfully! The chosen specifications are:")

    ######## Compute resulting sizes #######
    for key in required_keys:
        print("{}: {}".format(key, config[key]))

    ######## generate data #######

    # Define further configurations and initiate feature dict for storing restults
    path = (
        PosixPath(config["offline_data_path"])
        if PosixPath(config["offline_data_path"]).exists()
        else config["online_data_path"]
    )

    song_paths = [str(song_file) for song_file in list(path.rglob("*.wav"))]
    assert len(song_paths) > 0

    features = {}
    # Loop over all songs
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
            audio_wave, from_sr=sr, to_sr=config["sample_rate"], debug=config["verbose"]
        )
        audio_wave = mix_down(audio_wave)

        if config["beat_split"]:
            # Splitting track by beats
            _, beat_locations = get_beat_locations(
                audio_wave, sr=config["sample_rate"], verbose=config["verbose"]
            )
            # sequence audio file
            audio_seq = split_by_beats(
                audio_wave,
                beat_locations,
                idx,
                k_beats=config["hop_size"],
                num_samples=config["num_samples"],
                verbose=config["verbose"],
            )

        else:
            audio_seq = split_by_time(
                audio_wave,
                k_seconds=config["hop_size"] // 2,
                same_rate=config["sample_rate"],
            )

            features[str(idx)] = audio_seq

    # store data in hdf5 format in data directory
    data_path = Path(__file__).parent / "data"
    data_path.mkdir(exist_ok=True)

    with h5py.File(
        data_path / f"techno_{config['num_samples']}.h5",
        "w",
    ) as f:
        grp = f.create_group(f"{config['num_samples']}")
        for idx, track in features.items():
            grp.create_dataset(idx, data=track.numpy())
