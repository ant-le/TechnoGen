import sys, yaml, torch

sys.path.append(".")

from training.utils import make_model


def reconstruct(signal, model_name):
    with open(f"config/{model_name}", "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    model, _, _, _ = make_model(config, device, train=False)
    out, _, _ = model(signal)
    return out


if __name__ == "__main__":
    reconstruct()
