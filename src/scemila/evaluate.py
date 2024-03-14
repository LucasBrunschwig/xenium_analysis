import torch
import os
import json
from pathlib import Path
import pickle

from torchvision.transforms import transforms
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

from src.utils import get_results_path, check_gpu
from src.scemila.model import ImageClassificationModel


def confusion_matrix(model_, x_test_, y_test_):
    """ Build the confusion matrix from the model and grounds truth

    :param model_:
    :param x_test_:
    :param y_test_:
    :return:
    """

    raise NotImplementedError()


def explainability(model_, x_test_, method=""):
    """

    :return:
    """

    raise NotImplementedError()


def build_dir():
    dir_ = "nucleus_classification/"
    os.makedirs(dir_, exist_ok=True)
    return Path(dir_)


if __name__ == "__main__":

    # files that are assumed to be inside the training directory:
    # training.log - plot the visualization curves
    # training_params.json - the params that are needed to create the model
    # model_params.pth - the params that needs to be load from inside the model
    # parameters.pth - the parameters of the pretrained model to load inside the model

    training_name = "model_1e-5_100_224_vit_16"

    # ---------------------------------------------------------------------- #
    # Load Predefined Information

    # General setup
    results_dir = build_dir()
    model_dir = results_dir / training_name
    device = check_gpu()

    # Extract training and model params
    with open(model_dir / "training_params.json") as file:
        training_params = json.load(file)
    with open(model_dir / "model_params.json") as file:
        model_params = json.load(file)

    # Build the model based on the params
    model = ImageClassificationModel(**model_params)
    model.load(model_dir)

    # Load test dataset for evaluation
    dataset_path = get_results_path() / "scemila" / "stardist_qupath_he_dapi_match_leiden_clustering" / "dataset_test.pkl"
    with open(dataset_path, "rb") as file:
        X_test, y_test = pickle.load(file)
    y_test = torch.Tensor(y_test)

    # Build preprocess with same value as pretrained params
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((training_params["size"], training_params["size"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # ---------------------------------------------------------------------- #
    # Evaluate the models in different metrics

