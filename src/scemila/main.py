# TODO: change the preprocess and data augmentation -> model training for easier maintenance

# Std
import argparse
import pickle
import json
import os
from datetime import date
from pathlib import Path

# Third Party
import numpy as np
import torch
from torchvision import transforms
from loguru import logger

# Relative imports
from src.utils import get_results_path, check_gpu
from src.scemila.models import ImageClassificationModel
from src.scemila.train import ImageClassificationTraining
from src.scemila.utils import clear_directory, set_up_logger, set_up_logger_optuna, read_json, save_json_params
from src.scemila.hyperparameter_optimization import optuna_optimization


def hyperparameters_optimization(x_, y_, num_class_, preprocess_, augmentation_, model_type_, dataset_name_, size_,
                                 save_dir_, device_):

    # Build Directory
    optuna_dir = save_dir_ / "optuna"
    os.makedirs(optuna_dir, exist_ok=True)
    study_name = f"{date.today()}+{model_type_}+{dataset_name_}"
    save_study = optuna_dir / study_name
    os.makedirs(save_study, exist_ok=True)
    clear_directory(save_study)

    set_up_logger_optuna(save_study)

    optuna_study = {
        "sample": 2048,
        "metrics": ["balanced_accuracy", "class_accuracy"],
        "optimization": "balanced_accuracy"
    }

    model_params_definition = {
        "num_classes": num_class_,
        "in_dim": size_,
        "model_type": model_type_,
        "attention_layer": True,
        "unfrozen_layers": [[1, 4, 1], "int"],
        "n_layer_classifier": [[1, 3, 1], "int"],
    }

    training_params_definition = {
        # Fixed Arguments
        "preprocess": preprocess_,
        "transforms": augmentation_,
        "early_stopping": True,
        "results_dir": optuna_dir,
        "n_iter_min": 10,
        "n_iter_print": 1,
        "n_iter": 200,
        # Optimization Parameters
        "batch_size": [[0, 2, 1, 128, 2], "power_int"],
        "patience": [[1, 11, 5, 1], "multiply_int"],
        "lr": [[-7, -4, 1, 1, 10], "power_int"],
        "clipping_value": [[-1, 1, 1, 1, 10], "power_int"],
        "weight_decay": [[-5, -3, 1, 1, 10], "power_int"],
        "weighted_ce": [[True, False], "categorical"]
    }

    optuna_optimization(optuna_study, ImageClassificationModel, ImageClassificationTraining, x_, y_,
                        model_params_definition, training_params_definition, save_study, study_name, device_)


def simple_training(x_, y_, model_params_, training_params_, preprocess_, transforms_, dataset_name_, device_):
    """ Perform training with given parameters """

    # Build dir and Logger for the model
    lr_, n_iter_ = (training_params_[key] for key in ["lr", "n_iter"])
    model_type_, size_ = (model_params_[key] for key in ["model_type", "in_dim"])

    model_name = f"{date.today()}+{model_type_}+{dataset_name_}+{lr_}_{n_iter_}_{size_}"
    model_dir = results_dir / "simple_training" / model_name
    os.makedirs(model_dir, exist_ok=True)
    set_up_logger(model_dir)
    training_params_["results_dir"] = str(model_dir)

    # saving model params
    save_json_params(training_params_, model_dir / "training_params.json")
    save_json_params(model_params_, model_dir / "model_params.json")

    logger.info(f"Starting training and saving info to: {model_name}")

    model = ImageClassificationModel(**model_params)

    training = ImageClassificationTraining(model, preprocess=preprocess_, transforms=transforms_, device=device_,
                                           **training_params)

    training.train_(x_, y_)

    logger.info("Saving Model: to model_parameters.pth")
    model.save(model_dir)

    return model, training


def main(dataset_name_, model_params_, training_params_, optuna_, device_, results_dir_):

    # Load the corresponding DataSet
    dataset_path = get_results_path() / "scemila" / "datasets" / dataset_name_
    with open(dataset_path, "rb") as file:
        x_, y_ = pickle.load(file)

    # Extract the number of classes
    num_class_ = len(np.unique(y_))
    y_ = torch.Tensor(y_)

    model_params_["num_classes"] = num_class_

    # Build Preprocess and Data Augmentation transforms
    preprocess_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((model_params_["in_dim"], model_params_["in_dim"])),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    augmentation_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomVerticalFlip(0.3),
        transforms.RandomHorizontalFlip(0.3)
    ])

    if optuna_:
        hyperparameters_optimization(x_, y_, num_class_, preprocess_, augmentation_,
                                     model_params_["model_type"],
                                     dataset_name_,
                                     model_params_["in_dim"],
                                     results_dir_,
                                     device_)
    else:
        simple_training(x_, y_, model_params_, training_params_, preprocess_, augmentation_, dataset_name_, device_)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for learning rate and number of iterations")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of iterations for training")
    parser.add_argument("--size", type=int, default=128, help="Number of iterations for training")
    parser.add_argument("--unfrozen_layers", type=int, default=0, help="Number of unfrozen layers")
    parser.add_argument("--attention_layer", type=bool, default=True, help="Number of unfrozen layers")
    parser.add_argument("--model", type=str, default="resnet", help="[resnet, conv, vit]")
    parser.add_argument("--dataset", type=str, default=None, help="pickle file")
    parser.add_argument("--gpu", type=int, default=0, help="gpu number")
    parser.add_argument("--patience", type=int, default=10, help="patience number")
    parser.add_argument("--optuna", type=bool, default=False, help="whether use optuna to optimize the model")
    parser.add_argument("--config", type=str, default=None, help="configuration file")

    return parser.parse_args()


def build_dir():
    dir_ = "model_training"
    os.makedirs(dir_, exist_ok=True)
    return Path(dir_)


if __name__ == "__main__":

    # Parse arguments
    args = parse_arguments()
    lr = float(args.lr)
    n_iter = int(args.n_iter)
    preprocess_size = int(args.size)
    model_type = str(args.model)
    dataset_name = str(args.dataset)
    gpu_number = int(args.gpu)
    patience = int(args.patience)
    unfrozen_layers = int(args.unfrozen_layers)
    attention_layer = int(args.attention_layer)
    optuna = bool(args.optuna)
    config_file = args.config

    # build training directory
    results_dir = build_dir()

    # Extract global device
    device = check_gpu(gpu_number)

    model_params = {"num_classes": None,
                    "in_dim": preprocess_size,
                    "model_type": model_type,
                    "attention_layer": attention_layer,
                    "unfrozen_layers": unfrozen_layers,
                    "n_layer_classifier": 2,
                    }

    # Set up the parameter dicts
    training_params = {"batch_size": 256,
                       "lr": lr,
                       "n_iter": n_iter,
                       "n_iter_min": 10,
                       "early_stopping": True,
                       "n_iter_print": 1,
                       "patience": patience,
                       "clipping_value": 1.0,
                       "weight_decay": 1e-4,
                       "weighted_ce": True
                       }

    data_augmentation_params = {"rotation_angle": 10,
                                "vertical_flipping": 0.3,
                                "horizontal_flipping": 0.3,
                               }

    if config_file is not None:
        config_file = Path(config_file)
        training_params = read_json(config_file / "training_params.json")
        model_params = read_json(config_file / "model_params.json")

    # Call the main function
    main(dataset_name, model_params, training_params, optuna, device, results_dir)

