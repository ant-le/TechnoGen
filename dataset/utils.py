import torchaudio, torch, tqdm, sys

sys.path.append(".")


def resample(
    signal: torch.Tensor, from_sr: int, to_sr: int, debug=False
) -> torch.Tensor:
    """Resamples all audio files to a given sampling rate defined
    in the configuration file"""
    if from_sr != to_sr:
        resampler = torchaudio.transforms.Resample(from_sr, to_sr)
        signal = resampler(signal)
        if debug:
            tqdm.write(f"Track was resampled from original sr={from_sr}")
    return signal


def mix_down(signal: torch.Tensor) -> torch.Tensor:
    """Mixes down stereo signals to audio if necessary"""
    if signal.shape[0] > 1:
        # (n_channels, n_samples) -> (1, n_samples)
        signal = torch.mean(signal, dim=0, keepdim=True)
    return signal


def split_by_time(signal: torch.Tensor, k_seconds: int, same_rate: int):
    num_samples = k_seconds * same_rate

    if signal.shape[1] > num_samples:
        signal = torch.split(signal, num_samples, dim=1)

        # drop ending of the song due to different size
        signal = torch.stack(list(signal[:-1]), dim=0)

        # (split, 1, num_samples) -> (split, num_samples)
        signal = signal.reshape(signal.shape[0], signal.shape[2])

    elif signal.shape[1] < num_samples:
        signal = torch.nn.functional.pad(signal, (0, num_samples - signal.shape[1]))

    return signal
