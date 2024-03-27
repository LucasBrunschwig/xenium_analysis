""" The goal here is to perform modern data cleaning"""

import copy
import pickle
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from selfclean import SelfClean
from selfclean.cleaner.selfclean import PretrainingType

from src.utils import get_results_path


class MyCustomDataset(Dataset):
    def __init__(self, x, y, transform=None):
        """
        Args:
            x (list): List of features.
            y (list): List of labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = self.x[idx], self.y[idx]
        return self.x[idx], self.y[idx]


def data_cleaning(dataset_, dataset_name_, results_dir_):
    selfclean = SelfClean(
        plot_top_N=7,
    )
    issues = selfclean.run_on_dataset(
        dataset=copy.copy(dataset_),
        pretraining_type=PretrainingType.DINO,
        num_workers=os.cpu_count(),
        epochs=10,
        batch_size=16,
        save_every_n_epochs=1,
        dataset_name=dataset_name_,
        work_dir=results_dir_,
    )

    return issues

def build_dir():
    dir_ = get_results_path() / "data_cleaning"
    os.makedirs(dir_, exist_ok=True)

    return dir_


if __name__ == "__main__":
    print("Running Data Cleaning For Patches of Nuclei")

    dataset_name = "stardist_qupath_patch-HE_iou-0.5-1.0_pca-100_neigh_30_train.pkl"
    dataset_path = get_results_path() / "scemila" / "datasets" / dataset_name
    with open(dataset_path, "rb") as file:
        x, y = pickle.load(file)

    x = x[0:1000]
    y = y[0:1000]

    results_dir = build_dir()
    dataset = MyCustomDataset(x, y)
    data_cleaning(dataset, dataset_name, results_dir)


