from pytorch_lightning import LightningDataModule
from omegaconf import DictConfig
from os import cpu_count
from os.path import join, basename
import torch
from datas.samples import IDGBatch, IDGSample
from datas.datasets import IDGDataset
from typing import List, Optional
from torch.utils.data import DataLoader, Dataset
from vocabulary import Vocabulary
from datas.data_loader import dataloader
from transformers import AutoTokenizer


class IDGDataModule(LightningDataModule):
    # def __init__(self, config: DictConfig, vocab: Vocabulary):
    #     super().__init__()
    #     self.__vocab = vocab
    #     self.__config = config
    #     self.__data_folder = join(config.data_folder, config.dataset.name)
    #     self.__n_workers = cpu_count() if self.__config.num_workers == -1 else self.__config.num_workers
    #     self.__tokenizer = AutoTokenizer.from_pretrained(config.encoder.path)
    #     self.train_dataloader()
    
    def __init__(self, config: DictConfig):
        super().__init__()
        self.__config = config
        self.__data_folder = join(config.data_folder, config.dataset.name)
        self.__n_workers = cpu_count() if self.__config.num_workers == -1 else self.__config.num_workers
        self.__tokenizer = AutoTokenizer.from_pretrained(config.encoder.path)
        self.train_dataloader()

    @staticmethod
    def collate_wrapper(batch: List[IDGSample]) -> IDGBatch:
        return IDGBatch(batch)

    def __create_dataset(self, data_path: str) -> Dataset:
        # return IDGDataset(data_path, self.__config, self.__vocab, self.__tokenizer)
        return IDGDataset(data_path, self.__config, self.__tokenizer)

    def train_dataloader(self) -> DataLoader:
        train_dataset_path = join(self.__data_folder, basename(self.__data_folder)+"_train.json")
        train_dataset = self.__create_dataset(train_dataset_path)
        return DataLoader(
            train_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=self.__config.hyper_parameters.shuffle_data,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
            # persistent_workers=True
        )

    def val_dataloader(self) -> DataLoader:
        val_dataset_path = join(self.__data_folder, basename(self.__data_folder)+"_valid.json")
        val_dataset = self.__create_dataset(val_dataset_path)
        return DataLoader(
            val_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
            # persistent_workers=True
        )

    def test_dataloader(self) -> DataLoader:
        test_dataset_path = join(self.__data_folder, basename(self.__data_folder)+"_test.json")
        test_dataset = self.__create_dataset(test_dataset_path)
        return DataLoader(
            test_dataset,
            batch_size=self.__config.hyper_parameters.batch_size,
            shuffle=False,
            num_workers=self.__n_workers,
            collate_fn=self.collate_wrapper,
            pin_memory=True,
            # persistent_workers=True
        )

    def transfer_batch_to_device(self, batch: IDGBatch, device: Optional[torch.device] = None) -> IDGBatch:
        if device is not None:
            batch.move_to_device(device)
        return batch
