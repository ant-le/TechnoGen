from torch.utils.data import Dataset, BatchSampler, RandomSampler, DataLoader
import torch
from files_dataset import TechnoGenDataset


class HelperDataset(Dataset):
    def __init__(self, dataset, start, end):
        super().__init__()
        self.dataset = dataset
        self.start = start
        self.end = end
        assert 0 <= self.start < self.end <= len(self.dataset)

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, item):
        return self.dataset.get_item(self.start + item)


class DataProcessor:
    def __init__(self, config):
        self.dataset = TechnoGenDataset(config["dataset"])
        config = config["training"]
        self.create_datasets(config)
        self.create_samplers(config)
        self.create_data_loaders(config)
        self.print_stats()

    def set_epoch(self, config):
        self.train_sampler.set_epoch(config["epoch"])
        self.valid_sampler.set_epoch(config["epoch"])
        self.test_sampler.set_epoch(config["epoch"])

    def create_datasets(self, config):
        assert sum(config["split"]) == 1
        train_len = int(len(self.dataset) * config["split"][0])
        valid_len = int(len(self.dataset) * (config["split"][0] + config["split"][1]))
        self.train_dataset = HelperDataset(self.dataset, 0, train_len)
        self.valid_dataset = HelperDataset(self.dataset, train_len, valid_len)
        self.test_dataset = HelperDataset(self.dataset, valid_len, len(self.dataset))

    def create_samplers(self, config):
        self.train_sampler = BatchSampler(
            RandomSampler(self.train_dataset),
            batch_size=config["batch_size"],
            drop_last=True,
        )
        self.valid_sampler = BatchSampler(
            RandomSampler(self.valid_dataset),
            batch_size=config["batch_size"],
            drop_last=True,
        )
        self.test_sampler = BatchSampler(
            RandomSampler(self.test_dataset),
            batch_size=config["batch_size"],
            drop_last=True,
        )

    def create_data_loaders(self, config):
        # Loader to load mini-batches
        collate_fn = lambda batch: torch.stack([torch.from_numpy(b) for b in batch], 0)

        print("Creating Data Loader")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            # sampler=self.train_sampler,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.valid_loader = DataLoader(
            self.test_dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            # sampler=self.valid_sampler,
            pin_memory=False,
            drop_last=True,
            collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            # sampler=self.test_sampler,
            pin_memory=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

    def print_stats(self):
        print(
            f"Train {len(self.train_dataset)} samples. Valid {len(self.valid_dataset)} samples. Test {len(self.test_dataset)} samples"
        )
        print(f"Train sampler: {self.train_sampler}")
        print(f"Train loader: {len(self.train_loader)}")
