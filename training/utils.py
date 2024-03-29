import torch, sys

sys.path.append(".")

from pathlib import PosixPath

from model.vqvae.vqvae import VQVAE


def make_model(config, device, optimizer: str = "adam", train: bool = True):
    model_conf = config["model"]
    training_conf = config["training"]

    vqvae = VQVAE(
        input_shape=(
            (
                config["dataset"]["channels"],
                config["dataset"]["sample_rate"] * config["dataset"]["hop_size"],
            )
        ),
        layers=model_conf["layers"],
        stride=model_conf["stride"],
        width=model_conf["width"],
        depth=model_conf["depth"],
        codebook_dim=model_conf["codebook_dim"],
        codebook_size=model_conf["codebook_size"],
        discard_vec_threshold=model_conf["discard_vec_threshold"],
        codebook_loss_weight=model_conf["codebook_loss_weight"],
        spectral_loss_weight=model_conf["spectral_loss_weight"],
        commit_loss_weight=model_conf["commit_loss_weight"],
        init_random=model_conf["init_random"],
        lstm=model_conf["lstm"],
    )
    vqvae = vqvae.to(device)

    path = PosixPath("model", "vqvae", "parameter", f"{training_conf['name']}.pth")

    optimizer = get_optimizer(vqvae, training_conf)
    if path.exists():
        vqvae, optimizer, epoch = load_checkpoint(vqvae, optimizer, path, device)
        print("--- Model parameters were found and loaded successfully!")

    else:
        epoch = 0
        print("--- No model parameters were found!")
    if not train:
        vqvae.eval()
        for params in vqvae.parameters():
            params.requires_grad = False
    else:
        print("--- Start training from Epoch {}".format(epoch))
    return vqvae, optimizer, epoch


def save_checkpoint(model, optimizer, epoch, config):
    para_dir = PosixPath(
        "model", "vqvae", "parameter", f"{config['training']['name']}.pth"
    )
    para_dir.parent.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        str(para_dir),
    )


def load_checkpoint(model, optimizer, para_dir, device):
    checkpoint = torch.load(para_dir, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"] + 1

    return model, optimizer, epoch


def get_optimizer(model, config):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
    )
    if config["optimizer"] != "adam":
        print("Warning: Currently training is only supported with 'adam'!")
    return optimizer
