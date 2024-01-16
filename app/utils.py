import torchaudio, torch, h5py, sys

sys.path.append(".")


from dataset.utils import resample, mix_down, split_by_time, augment
from training.utils import make_model

# define used model and parameters here


def get_model(config):
    model, _, _ = make_model(config, torch.device("cpu"), train=False)
    return model, config["dataset"]


def encode(config):
    model, config = get_model(config)
    audio, sr = torchaudio.load(config["path"] / "input.wav")
    audio = resample(audio, sr, config["sample_rate"])
    audio = mix_down(audio)
    audio_splits = split_by_time(audio, config["hop_size"], config["sample_rate"])
    audio_splits = audio_splits.view(audio_splits.shape[0], 1, audio_splits.shape[1])
    audio_encoded = model.encode(audio_splits)

    with h5py.File(
        config["path"] / "embedding.h5",
        "w",
    ) as f:
        f.create_dataset("embedding", data=audio_encoded.numpy())


def decode(config):
    model, config = get_model(config)
    with h5py.File(
        config["path"] / "embedding.h5",
        "r",
    ) as f:
        embedding = torch.Tensor(f["embedding"])
    out = model.decode(embedding)
    out = out.reshape(1, out.shape[0] * out.shape[2])
    torchaudio.save(
        config["path"] / "reconstructed.wav", out, config["sample_rate"], format="wav"
    )


def sample(config):
    model, config = get_model(config)
    out = model.generate()
    out = out.reshape(1, out.shape[2])
    out = augment(out, sr=config["sample_rate"])
    torchaudio.save(
        config["path"] / "generated.wav",
        out,
        config["sample_rate"],
        format="wav",
    )


def refresh(config):
    for name in ["embedding.h5", "input.wav", "reconstructed.wav"]:
        file = config["dataset"]["path"] / name
        file.unlink(missing_ok=True)
