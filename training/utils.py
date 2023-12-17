from model.vqvae.vqvae import VQVAE

from pathlib import PosixPath
import torch


def make_model(config, device, optimizer: str = "adam", train: bool = True):
    model_conf = config["model"]
    training_conf = config["training"]

    vqvae = VQVAE(
        input_shape=((config["dataset"]["channels"], config["dataset"]["num_samples"])),
        layers=model_conf["layers"],
        kernel_size=model_conf["kernel_size"],
        stride=model_conf["stride"],
        width=model_conf["width"],
        depth=model_conf["depth"],
        codebook_dim=model_conf["codebook_dim"],
        codebook_size=model_conf["codebook_size"],
        discard_vec_threshold=model_conf["discard_vec_threshold"],
        codebook_loss_weight=model_conf["codebook_loss_weight"],
        spectral_loss_weight=model_conf["spectral_loss_weight"],
        commit_loss_weight=model_conf["commit_loss_weight"],
    )
    vqvae = vqvae.to(device)

    path = PosixPath("model", "vqvae", "parameter", f"{training_conf['name']}.pth")

    optimizer = get_optimizer(vqvae, training_conf)
    if path.exists():
        vqvae, optimizer, epoch, benchmarks = load_checkpoint(vqvae, optimizer, path)
        print(
            f"--- Model parameters loaded successfully -> Continue with epoch {epoch}"
        )
    else:
        epoch = 0
        benchmarks = {  # TODO: modify to dynamically write dict
            "train": {
                "loss": [],
                "recons_loss": [],
                "spectral_loss": [],
                "commit_loss": [],
            },
            "valid": {"loss": []},
            "test": {"loss": []},
        }
        print(f"--- New model parameters -> Start with new model from epoch {epoch}")
    if not train:
        vqvae.eval()
        for params in vqvae.parameters():
            params.requires_grad = False
    return vqvae, optimizer, epoch, benchmarks


def save_checkpoint(model, optimizer, epoch, benchmarks, config):
    path = str(
        PosixPath("model", "vqvae", "parameter", f"{config['training']['name']}.pth")
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "benchmarks": benchmarks,
        },
        path,
    )


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(str(load_path))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    benchmarks = checkpoint["benchmarks"]
    epoch = checkpoint["epoch"]

    return model, optimizer, epoch, benchmarks


def get_optimizer(model, config):
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["lr"],
        )
    else:
        print("Currently training is only supported with 'adam'")
    return optimizer
