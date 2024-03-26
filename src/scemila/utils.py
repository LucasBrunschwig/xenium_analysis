import json
import os
import pickle

from loguru import logger
import sys

def check_training_params():
    """ Check that all the parameters are defined """
    pass

def check_model_params():
    """ Check that all the parameters are defined """
    pass


def read_json(filename):
    with open(filename, "r") as file:
        dict_ = json.load(file)
    return dict_


def clear_directory(directory_path):
    try:
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except OSError:
        print("Error occurred while deleting files.")


def filter_loss_training(record):
    return record["message"].startswith("loss::")


def filter_out_loss_training(record):
    return not record["message"].startswith("loss::")


def save_json_params(dict_, file_):
    with open(file_, "w") as file:
        json.dump(dict_, file)


def save_pickle_params(dict_, file_):
    with open(file_, "wb") as file:
        pickle.dump(dict_, file)

def set_up_logger(model_dir):
    for handler_id in logger._core.handlers:
        logger.remove(handler_id)
    logger.add(sys.stdout, format="{time:YYYY:MM:DD:HH:mm} | {level} | {message}", level="INFO",
               filter=filter_out_loss_training, colorize=True)
    logger.add(f"{model_dir}/file.log", format="{message}", level="INFO", filter=filter_out_loss_training)
    logger.add(f"{model_dir}/train.log", format="{message}", level="INFO", filter=filter_loss_training)


def set_up_logger_optuna(optuna_dir):
    for handler_id in logger._core.handlers:
        logger.remove(handler_id)
    logger.add(f"{optuna_dir}/file.log", format="{message}", level="INFO", filter=filter_out_loss_training)