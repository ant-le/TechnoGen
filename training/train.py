import argparse, yaml, wandb, sys, os

os.environ["WANDB_SILENT"] = "true"

sys.path.append(".")
from tqdm import tqdm
from pathlib import PosixPath

import torch
from torch.optim.lr_scheduler import MultiStepLR

from dataset.data_processor import DataProcessor
from training.utils import make_model, save_checkpoint


def train(model, data_processor, optimizer, device):
    losses = []
    metrics = {  # TODO: modify to dynamically write dict
        "recons_loss": [],
        "spectral_loss": [],
        "commit_loss": [],
    }

    model.train()

    for _, batch in enumerate(
        tqdm(data_processor, position=1, leave=False, desc="Batch Number")
    ):
        audio_batch = batch.to(device)

        # Feed foward pass
        optimizer.zero_grad()
        _, loss, loss_comps = model(audio_batch)
        losses.append(loss)

        for key in metrics.keys():
            metrics[key].append(loss_comps[key])

        # Propagate back loss and update gradients
        loss.backward()
        optimizer.step()

    # calculate average metrics for epochs
    model.eval()
    epoch_loss = sum(losses) / len(losses)
    metrics = {key: sum(values) / len(values) for key, values in metrics.items()}

    torch.cuda.empty_cache()

    return epoch_loss, metrics


def evaluate(model, data_processor, device):
    losses = []
    model.eval()
    for batch in data_processor:
        audio_batch = batch.to(device)

        with torch.no_grad():
            _, loss, _ = model(audio_batch)
            losses.append(loss)

    # calculate average metrics for epochs
    epoch_loss = sum(losses) / len(losses)

    return epoch_loss


def run_epochs(config, save_checkpoints: bool = True):
    best_loss = epochs_without_improve = 0

    # Loading data
    data_processor = DataProcessor(config["dataset"])

    # Loading model
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print("--- Model runs on device: {}".format(device.type))

    model, optimizer, epoch, benchmarks = make_model(config, device=device)

    # setting up adaptive learning rate
    scheduler = MultiStepLR(
        optimizer,
        milestones=[20, 50, 80],
        gamma=0.1,
        last_epoch=epoch if epoch > 0 else -1,
    )

    # Setup Logger
    wandb.login()
    wandb.init(
        config=config["training"],
        project="TechnoGen-training",
    )
    wandb.watch(model)

    # Define training loop
    for epoch in tqdm(
        range(epoch, config["training"]["epochs"]),
        position=0,
        leave=True,
        desc="Running Epochs",
    ):
        # train and validate model
        loss, metrics = train(model, data_processor.train_loader, optimizer, device)
        valid_loss = evaluate(model, data_processor.valid_loader, device)

        # Early stopping
        max_no_improvement = config["training"]["early_stop"]
        if valid_loss < best_loss:
            best_loss = valid_loss
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1

            if epochs_without_improve >= max_no_improvement:
                tqdm.write(
                    f"--- No improvements for {max_no_improvement} consequetive epochs: Early Stopping with best loss = {best_loss}"
                )
                break

        # Step for learning rate decay
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()),
            max_norm=config["training"]["clipping_fac"],
        )

        scheduler.step()

        # save model params every 5 epochs
        if epoch % 5 == 0 and save_checkpoints:
            save_checkpoint(model, optimizer, epoch, benchmarks, config)

        # Loggging
        metrics.update(dict(train_loss=loss, valid_loss=valid_loss))
        wandb.log(metrics)
        tqdm.write(
            f"--- Epoch finished with Training Loss: {loss} and valid Loss: {valid_loss}"
        )

    # logging test accuracy
    test_loss = evaluate(model, data_processor.test_loader, device)
    wandb.log({"Test Loss": test_loss})
    tqdm.write(f"--- Training finished with Test Loss: {test_loss}")


if __name__ == "__main__":
    ######## Read the config file #######
    parser = argparse.ArgumentParser(
        description="Configurate model and training specification."
    )
    parser.add_argument(
        "--config",
        default="config/baseline.yml",
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

    run_epochs(config=config)
