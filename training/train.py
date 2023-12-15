import argparse, yaml
from tqdm import tqdm
from pathlib import PosixPath

if __name__ == "__main__":
    ######## Read the config file #######
    parser = argparse.ArgumentParser(
        description="Configurate model and training specification."
    )
    parser.add_argument(
        "--config",
        default="config/train.yml",
        help="Path to config file",
        type=str,
    )
    args = parser.parse_args()
    if not PosixPath(args.config).exists():
        raise FileNotFoundError(f"Config file {args.config} does not exist.")

    with open(args.config, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    required_keys = [
        "model",
        "training",
    ]

    if not all(key in config for key in required_keys):
        raise ValueError(
            f"Config file must contain properties {','.join(required_keys)}."
        )
    print(
        "Configuration file loaded successfully! The chosen specifications are:\n {}".format(
            config
        )
    )
