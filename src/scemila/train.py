import argparse
import json
import os
from pathlib import Path

from loguru import logger
import torch
from torch import nn
import time
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import pickle
import sys

from tqdm import tqdm

from src.utils import check_gpu, get_results_path
from src.scemila.custom_dataset import TrainingImageDataset, TestImageDataset
from src.scemila.model import ImageClassificationModel
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = check_gpu()




class ImageClassificationTraining(nn.Module):
    def __init__(self, model, batch_size, lr, n_iter, n_iter_min, early_stopping, n_iter_print, patience,
                 preprocess, transforms, clipping_value, weight_decay, results_dir):
        super(ImageClassificationTraining, self).__init__()

        # Model
        self.model = model.to(DEVICE)

        # Training
        self.batch_size = batch_size
        self.lr = lr
        self.n_iter = n_iter
        self.n_iter_min = n_iter_min
        self.n_iter_print = n_iter_print
        self.patience = patience
        self.clipping_value = clipping_value
        self.early_stopping = early_stopping
        self.results_dir = results_dir

        # Create

        # Preprocessing and Data Augmentation
        self.preprocess = preprocess
        self.transforms = transforms

        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

    def train_(self, X: torch.Tensor, y: torch.Tensor):
        y = self._check_tensor(y).squeeze().long()

        dataset = TrainingImageDataset(
            X, y, preprocess=self.preprocess, transform=self.transforms
        )

        # Define the train-validation split ratio
        train_ratio = 0.85

        # Split the dataset while maintaining class distribution
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_ratio)
        train_indices, val_indices = next(splitter.split(dataset.targets, dataset.targets))

        # Create subsets for training and validation
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        # Create DataLoader for training and validation
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, prefetch_factor=3, num_workers=4, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, prefetch_factor=3, num_workers=4, drop_last=True)

        # do training
        val_loss_best = 999999
        patience = 0
        self.model.train()

        loss = nn.CrossEntropyLoss()

        for i in range(self.n_iter):
            train_loss = []
            start_ = time.time()
            progress_bar = tqdm(total=np.ceil(len(train_indices) / self.batch_size), desc="Processing Batch")
            for batch_ndx, sample in enumerate(train_loader):
                progress_bar.update(1)
                self.optimizer.zero_grad()

                X_next, y_next = sample
                X_next = X_next.to(DEVICE)
                y_next = y_next.to(DEVICE)

                preds = self.model.forward(X_next)

                batch_loss = loss(preds, y_next)

                batch_loss.backward()

                self.optimizer.step()

                train_loss.append(batch_loss.detach())

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clipping_value
                )

            progress_bar.close()
            train_loss = torch.Tensor(train_loss).to(DEVICE)

            if self.early_stopping or i % self.n_iter_print == 0:
                with torch.no_grad():

                    val_loss = []

                    for batch_val_ndx, sample in enumerate(val_loader):
                        X_val_next, y_val_next = sample
                        X_val_next = X_val_next.to(DEVICE)
                        y_val_next = y_val_next.to(DEVICE)

                        preds = self.model.forward(X_val_next)
                        val_loss.append(loss(preds, y_val_next).detach())

                val_loss = torch.mean(torch.Tensor(val_loss))

                end_ = time.time()

                train_loss_mean = torch.mean(train_loss)
                logger.info(f"loss::{i},{val_loss},{train_loss_mean}")

                if i % self.n_iter_print == 0:
                    logger.info(
                        f"Epoch: {i}, val_loss: {val_loss:.4f}, train_loss: {train_loss_mean:.4f}, epoch elapsed time: {(end_ - start_):.2f}")
                    # write this by default to get

                if self.early_stopping:
                    if val_loss_best > val_loss:
                        val_loss_best = val_loss
                        patience = 0
                    else:
                        patience += 1

                    if patience > self.patience and i > self.n_iter_min:
                        logger.info(
                            f"Final Epoch: {i}, val_loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}"
                        )
                        break

        return self

    def test(self, X):
        self.model.to(DEVICE)
        self.model.eval()
        with torch.no_grad():
            results = np.empty((0, 1))
            test_loader = DataLoader(TestImageDataset(X, preprocess=self.preprocess), batch_size=self.batch_size,
                                     pin_memory=False)

            for batch_test_ndx, X_test in enumerate(test_loader):
                results = np.vstack(
                    (
                        results,
                        np.expand_dims(
                            self.model(X_test.to(DEVICE))
                            .argmax(dim=-1)
                            .detach()
                            .cpu()
                            .numpy(),
                            axis=1,
                        ).astype(int),
                    )
                )
            return pd.DataFrame(results)

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.cpu()
        else:
            return torch.from_numpy(np.asarray(X)).cpu()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for learning rate and number of iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations for training")
    parser.add_argument("--size", type=int, default=128, help="Number of iterations for training")
    parser.add_argument("--model", type=str, default="resnet", help="[resnet, conv, vit]")
    parser.add_argument("--dataset", type=str, default=None, help="pickle file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")

    return parser.parse_args()


def build_dir():
    dir_ = "training"
    os.makedirs(dir_, exist_ok=True)
    return Path(dir_)


def filter_loss_training(record):
    return record["message"].startswith("loss::")


def filter_out_loss_training(record):
    return not record["message"].startswith("loss::")


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()
    lr = float(args.lr)
    n_iter = int(args.n_iter)
    size = int(args.size)
    model_type = str(args.model)
    dataset_name = str(args.dataset)
    gpu_number = int(args.gpu)

    training_params = {"lr": lr, "n_iter": n_iter, "size": size, "model_type": model_type}
    model_name = f"model_{lr}_{n_iter}_{size}_{model_type}+{dataset_name}"

    results_dir = build_dir()
    model_dir = results_dir / model_name
    os.makedirs(model_dir, exist_ok=True)

    for handler_id in logger._core.handlers:
        logger.remove(handler_id)
    logger.add(sys.stdout, format="{time:YYYY:MM:DD:HH:mm} | {level} | {message}", level="INFO", filter=filter_out_loss_training, colorize=True)
    logger.add(f"{model_dir}/file.log", format="{message}", level="INFO", filter=filter_out_loss_training)
    logger.add(f"{model_dir}/train.log", format="{message}", level="INFO", filter=filter_loss_training)

    logger.info(f"Starting training and saving info to: {model_name}")

    # saving model params
    with open(model_dir / "training_params.json", "w") as file:
        json.dump(training_params, file)

    # Load DataSet
    dataset_path = get_results_path() / "scemila" / "datasets" / dataset_name
    with open(dataset_path, "rb") as file:
        X, y = pickle.load(file)

    num_class = len(np.unique(y))
    y = torch.Tensor(y)

    device = torch.device(f"cuda:{gpu_number}")

    attention_layer = True
    model = ImageClassificationModel(num_classes=num_class, in_dim=size, model_type=model_type, attention_layer=attention_layer)
    model_params = {"num_classes": num_class, "in_dim": size, "model_type": model_type, "attention_layer": attention_layer}
    with open(model_dir / "model_params.json", "w") as file:
        json.dump(model_params, file)

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((size, size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomHorizontalFlip(0.3)
    ])

    training = ImageClassificationTraining(model,
                                           batch_size=128,
                                           lr=lr,
                                           n_iter=n_iter,
                                           n_iter_min=10,
                                           early_stopping=True,
                                           n_iter_print=1,
                                           patience=3,
                                           preprocess=preprocess,
                                           transforms=transforms,
                                           clipping_value=0.0,
                                           weight_decay=1e-4,
                                           results_dir=model_dir
                                           )

    training.train_(X, y)

    logger.info("Saving Model: to model_parameters.pth")
    model.save(model_dir)
