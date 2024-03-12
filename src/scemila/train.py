import argparse

import torch
from torch import nn
import time
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import pickle

from tqdm import tqdm

from src.utils import check_gpu, get_results_path
from src.scemila.custom_dataset import TrainingImageDataset, TestImageDataset
from src.scemila.model import ImageClassificationModel
from sklearn.model_selection import StratifiedShuffleSplit

DEVICE = check_gpu()


class ImageClassificationTraining(nn.Module):
    def __init__(self, model, batch_size, lr, n_iter, n_iter_min, early_stopping, n_iter_print, patience,
                 preprocess, transforms, clipping_value, weight_decay):
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

        # Preprocessing and Data Augmentation
        self.preprocess = preprocess
        self.transforms = transforms

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
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, prefetch_factor=3, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, prefetch_factor=3, num_workers=4)

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

                preds = self.model.forward(X_next).squeeze()

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

                        preds = self.model.forward(X_val_next).squeeze()
                        val_loss.append(loss(preds, y_val_next).detach())

                    val_loss = torch.mean(torch.Tensor(val_loss))

                    end_ = time.time()

                    if self.early_stopping:
                        if val_loss_best > val_loss:
                            val_loss_best = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience > self.patience and i > self.n_iter_min:
                            print(
                                f"Final Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}"
                            )
                            break

                    if i % self.n_iter_print == 0:
                        print(
                            f"Epoch: {i}, loss: {val_loss:.4f}, train_loss: {torch.mean(train_loss):.4f}, epoch elapsed time: {(end_ - start_):.2f}"
                        )

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

    def save(self):
        raise ValueError("TBD")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for learning rate and number of iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations for training")
    parser.add_argument("--size", type=int, default=128, help="Number of iterations for training")
    parser.add_argument("--model", type=str, default="resnet", help="[resnet, conv, vit]")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    lr = float(args.lr)
    n_iter = int(args.n_iter)
    size = int(args.size)
    model_type = str(args.model)

    # Load DataSet
    dataset_path = get_results_path() / "scemila" / "stardist_qupath_he_dapi_match_leiden_clustering" / "dataset.pkl"
    with open(dataset_path, "rb") as file:
        X, y = pickle.load(file)

    num_class = len(np.unique(y))
    y = torch.Tensor(y)

    device = check_gpu()
    model = ImageClassificationModel(num_classes=num_class, in_dim=size, model_type=model_type, attention_layer=True)

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

    training =  ImageClassificationTraining(model,
                                            batch_size=128,
                                            lr=lr,
                                            n_iter=n_iter,
                                            n_iter_min=100,
                                            early_stopping=True,
                                            n_iter_print=1,
                                            patience=10,
                                            preprocess=preprocess,
                                            transforms=transforms,
                                            clipping_value=0.0,
                                            weight_decay=1e-4
                                            )

    training.train_(X, y)
