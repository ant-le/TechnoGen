import h5py, torchaudio, sys

sys.path.append(".")

from tqdm import tqdm
from pathlib import PosixPath, Path

from dataset.utils import resample, mix_down, split_by_time


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

    # Define further configurations and initiate feature dict for storing results
    path = (
        PosixPath(config["offline_data_path"])
        if PosixPath(config["offline_data_path"]).exists()
        else config["online_data_path"]
    )

    # get paths of all audio files in directory
    song_paths = [str(song_file) for song_file in list(path.rglob("*.wav"))]
    # song_paths.extend([str(song_file) for song_file in list(path.rglob("*.mp3"))])
    print(len(song_paths))
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
