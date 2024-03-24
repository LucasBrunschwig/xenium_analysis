import pickle
from functools import partial

import numpy as np
from loguru import logger
import optuna
import torch
import random
from sklearn.model_selection import StratifiedKFold

from src.scemila.metrics_utils import ClassifierMetrics

best_val = None


def get_random_indices(labels, n_sample):

    label_indices = {}
    for index, label in enumerate(labels):
        label_int = int(label)
        if label_int in label_indices:
            label_indices[label_int].append(index)
        else:
            label_indices[label_int] = [index]

    random_indices = []
    for label, indices in label_indices.items():
        ix = random.sample(indices, min(n_sample, len(indices)))
        random_indices.extend(ix)

    return np.array(random_indices)


def build_params(params_collection, params_selection_str, trial):
    params_selection = {}
    for param_name, description in params_collection.items():
        if isinstance(description, list):
            range_ = description[0]
            distribution = description[1]
            if distribution == "log_uniform":
                params_selection[param_name] = trial.suggest_loguniform(param_name, range_[0], range_[1])
            elif distribution == "uniform":
                params_selection[param_name] = trial.suggest_uniform(param_name, range_[0], range_[1])
            elif distribution == "categorical":
                params_selection[param_name] = trial.suggest_categorical(param_name, range_)
            elif distribution == "int":
                params_selection[param_name] = trial.suggest_int(param_name, range_[0], range_[-1])
            params_selection_str += f"{param_name}-{params_selection[param_name]}, "
        else:
            params_selection[param_name] = description

    return params_selection, params_selection_str


def objective(trial, optuna_params, model_params, training_params, x, y, model, training, save_model):

    # Instantiate Parameters
    params_selection_str = f"Starting Trial {trial.number}: "
    params_tr, params_selection_str = build_params(training_params, params_selection_str, trial)
    params_mo, params_selection_str = build_params(model_params, params_selection_str, trial)

    # Parameters selection log
    print(params_selection_str)
    logger.info(params_selection_str)

    # Create Instances
    model_instance = model(**params_mo)
   #torch.compile(model_instance)
    params_tr["model"] = model_instance
    training_instance = training(**params_tr)

    stratifier = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)

    sampling = optuna_params["sample"]
    if sampling is not None:
        ix = get_random_indices(y, sampling)
        x = x[ix]
        y = y[ix]

    metrics = ClassifierMetrics(optuna_params["metrics"])
    metric_fold = {metric: [] for metric in optuna_params["metrics"]}
    train_fold = []
    val_fold = []
    for train_index, test_index in stratifier.split(x, y):

        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]

        train_loss, val_loss = training_instance.train_(x_train, y_train)
        train_fold.append(train_loss)
        val_fold.append(val_loss)

        preds = training_instance.predict_proba(x_test)
        scores = metrics.score_proba(y_test, preds)
        for name, score in scores.items():
            metric_fold[name].append(score)

    # Store loss
    for name, metric in metric_fold.items():
        if name != "class_accuracy":
            print(f"{name}: {np.mean(metric):.3f} +/- {np.std(metric):.3f}")
        else:
            str_ = f"{name}: "
            for label in metric[0].keys():
                label_values = [value[label] for value in metric]
                str_ += f"{label}: {np.mean(label_values):.2f} +/-  {np.std(label_values):.2f} |"
            print(str_)

    optimization_value = np.mean(metric_fold[optuna_params["optimization"]])

    global best_val
    if best_val is None or optimization_value < best_val:
        best_val = optimization_value
        model_instance.save(save_model)

    return optimization_value


def optuna_optimization(optuna_params, model, training, X, y, model_params, training_params, save_dir, study_name):

    with open(save_dir / "model_params.json", "wb") as file:
        pickle.dump(model_params, file)
    with open(save_dir / "training_params.json", "wb") as file:
        pickle.dump(training_params, file)

    # Create a study object and specify the optimization direction as 'minimize'.
    study = optuna.create_study(study_name=study_name, direction="minimize",
                                storage=f"sqlite:///{save_dir}/trial.db")

    logger.info(f"Starting study with: {study_name} - stored in {save_dir}/trial.db")

    # Start the optimization; the number of trials can be adjusted
    objective_with_params = partial(objective, model_params=model_params, training_params=training_params,
                                    x=X, y=y, model=model, training=training, save_model=save_dir, optuna_params=optuna_params)
    study.optimize(objective_with_params, n_trials=30)

    # Print the optimal parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # Optionally, you can print the best value (e.g., lowest validation loss)
    best_value = study.best_value
    print(f"Best value (lowest validation loss): {best_value}")