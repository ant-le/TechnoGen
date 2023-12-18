import torch
from torch.utils.data import Dataset, DataLoader
from dataset.files_dataset import TechnoGenDataset


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
        return self.dataset[self.start + item]


class DataProcessor:
    def __init__(self, config):
        self.dataset = TechnoGenDataset(config)
        self.create_datasets(config)
        self.create_data_loaders(config)
        self.print_stats()

    def create_datasets(self, config):
        assert sum(config["split"]) == 1
        train_len = int(len(self.dataset) * config["split"][0])
        valid_len = int(len(self.dataset) * (config["split"][0] + config["split"][1]))
        self.train_dataset = HelperDataset(self.dataset, 0, train_len)
        self.valid_dataset = HelperDataset(self.dataset, train_len, valid_len)
        self.test_dataset = HelperDataset(self.dataset, valid_len, len(self.dataset))

    def create_data_loaders(self, config):
        # Loader to load mini-batches
        collate_fn = lambda batch: torch.stack([torch.from_numpy(b) for b in batch], 0)

        print("--- Creating Data Loader")
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=1,
            # collate_fn=collate_fn,
        )
        self.valid_loader = DataLoader(
            self.test_dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=1,
            # collate_fn=collate_fn,
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config["batch_size"],
            shuffle=config["shuffle"],
            num_workers=1,
            # collate_fn=collate_fn,
        )

    def print_stats(self):
        print(
            f"--- Data Loader created with sizes: Train {len(self.train_dataset)} samples. Valid {len(self.valid_dataset)} samples. Test {len(self.test_dataset)} samples"
        )
