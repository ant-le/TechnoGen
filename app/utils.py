import torchaudio, torch, h5py, sys

sys.path.append(".")


from dataset.utils import resample, mix_down, split_by_time, augment
from model.vqvae.vqvae import VQVAE

# define used model and parameters here


def get_model():
    model = VQVAE()
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    model.to(device)
    checkpoint = torch.hub.load_state_dict_from_url(
        "https://github.com/ant-le/TechnoGen/releases/download/v1.0.0/baseline.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    return model, device


def encode(config):
    model, device = get_model()
    audio, sr = torchaudio.load(config["path"] / "input.wav")
    audio = resample(audio, sr, config["sample_rate"])
    audio = mix_down(audio)
    audio_splits = split_by_time(audio, config["hop_size"], config["sample_rate"])
    audio_splits = audio_splits.to(device)
    audio_splits = audio_splits.view(audio_splits.shape[0], 1, audio_splits.shape[1])
    audio_encoded = model.encode(audio_splits)

    with h5py.File(
        config["path"] / "embedding.h5",
        "w",
    ) as f:
        f.create_dataset("embedding", data=audio_encoded.cpu().numpy())


def decode(config):
    model, device = get_model()
    with h5py.File(
        config["path"] / "embedding.h5",
        "r",
    ) as f:
        embedding = torch.Tensor(f["embedding"]).to(device)
    out = model.decode(embedding)
    out = out.reshape(1, out.shape[0] * out.shape[2]).cpu()
    torchaudio.save(
        config["path"] / "reconstructed.wav",
        out,
        config["sample_rate"],
        format="wav",
    )


def sample(config):
    model, device = get_model()
    out = model.generate()
    out = out.reshape(1, out.shape[2]).cpu()
    out = augment(out, sr=config["sample_rate"])
    torchaudio.save(
        config["path"] / "generated.wav",
        out,
        config["sample_rate"],
        format="wav",
    )


def refresh(config):
    for name in ["embedding.h5", "input.wav", "reconstructed.wav"]:
        file = config["path"] / name
        file.unlink(missing_ok=True)
