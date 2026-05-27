# Copyright 2024 Haowen Yu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
from dataclasses import asdict

import lightning.pytorch as pl
from torch.utils.data import DataLoader

from configs.sections import DataConfig


class DataInterface(pl.LightningDataModule):
    """LightningDataModule wrapper that wires custom datasets based on grouped config sections."""

    def __init__(self, data_cfg: DataConfig):
        super().__init__()
        self.data_cfg = data_cfg
        self.save_hyperparameters({"data": asdict(self.data_cfg)})

        self.dataset_cfg = self.data_cfg.dataset
        self.dataloader_cfg = self.data_cfg.dataloader

        self.train_set, self.validation_set, self.test_set = self.__load_data_module()

    # Lightning hook function, override to implement loading train dataset
    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_set,
            batch_size=self.dataloader_cfg.batch_size,
            num_workers=self.dataloader_cfg.num_workers,
            shuffle=self.dataloader_cfg.shuffle_train,
            persistent_workers=self.dataloader_cfg.persistent_workers,
            pin_memory=self.dataloader_cfg.pin_memory,
            multiprocessing_context=self.dataloader_cfg.multiprocessing_context,
            drop_last=self.dataloader_cfg.drop_last,
        )

    # Lightning hook function, override to implement loading validation dataset
    def val_dataloader(self):
        test_batch = self.dataloader_cfg.test_batch_size or self.dataloader_cfg.batch_size
        return DataLoader(
            dataset=self.validation_set,
            batch_size=test_batch,
            num_workers=self.dataloader_cfg.num_workers,
            shuffle=self.dataloader_cfg.shuffle_val,
            persistent_workers=self.dataloader_cfg.persistent_workers,
            pin_memory=self.dataloader_cfg.pin_memory,
            multiprocessing_context=self.dataloader_cfg.multiprocessing_context,
            drop_last=self.dataloader_cfg.drop_last,
        )

    # Lightning hook function, override to implement loading test dataset
    def test_dataloader(self):
        test_batch = self.dataloader_cfg.test_batch_size or self.dataloader_cfg.batch_size
        return DataLoader(
            dataset=self.test_set,
            batch_size=test_batch,
            num_workers=self.dataloader_cfg.num_workers,
            shuffle=self.dataloader_cfg.shuffle_test,
            persistent_workers=self.dataloader_cfg.persistent_workers,
            pin_memory=self.dataloader_cfg.pin_memory,
            multiprocessing_context=self.dataloader_cfg.multiprocessing_context,
            drop_last=self.dataloader_cfg.drop_last,
        )
    
    @staticmethod
    def filter_init_args(cls, config_dict):
        """
        Checks if config_dict has all required arguments for cls.__init__
        """
        init_args = dict()
        for name in inspect.signature(cls.__init__).parameters.keys():
            # Skip 'self', '*args', '**kwargs' and parameters with defaults
            if name not in ('self', 'purpose'):
                init_args[name] = config_dict[name]
        provided_keys = set(config_dict.keys())
        missing_keys = init_args.keys() - provided_keys
        
        if missing_keys:
            raise ValueError(f"In dataset initialization, found missing config keys for {cls.__name__}: {missing_keys}")
        
        return init_args

    def __load_data_module(self):
        file_name = self.dataset_cfg.file_name
        class_name = self.dataset_cfg.class_name
        if class_name is None:
            raise ValueError("DATA.dataset.class_name must be specified in the configuration.")
        if file_name is None:
            raise ValueError("DATA.dataset_file_name must be specified in the configuration.")
        try:
            data_class = getattr(importlib.import_module('data.' + file_name, package=__package__), class_name)
        except Exception:
            raise ValueError(f'Invalid Dataset File Name data.{file_name} or Invalid Class Name data.{file_name}.{class_name}')

        dataset_kwargs = self.dataset_cfg.dataset_init_args
        # Only validate highest level keyword arguments. This is a tradeoff between flexibility and rigour.
        # If you want to enable recursive validation for every keyword including nested ones, define them as template 
        # in config schema instead of using raw dictionary.
        # We assume that dataset_kwargs is a superset of data_class's init arg set.
        filtered_dataset_kwargs = self.filter_init_args(cls=data_class, config_dict=dataset_kwargs)
        train_set = data_class(**filtered_dataset_kwargs, purpose='train')
        validation_set = data_class(**filtered_dataset_kwargs, purpose='validation')
        test_set = data_class(**filtered_dataset_kwargs, purpose='test')

        return train_set, validation_set, test_set
