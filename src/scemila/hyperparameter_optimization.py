import pickle
from functools import partial

import numpy as np
from loguru import logger
import optuna
import random
from sklearn.model_selection import StratifiedKFold
import torch

from src.scemila.metrics_utils import ClassifierMetrics
from src.scemila.utils import save_pickle_params


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
            elif distribution == "log_float":
                params_selection[param_name] = trial.suggest_float(param_name, range_[0], range_[1], log=True)
            elif distribution == "log_int":
                params_selection[param_name] = trial.suggest_int(param_name, range_[0], range_[1], log=True)
            elif distribution == "float":
                params_selection[param_name] = trial.suggest_float(param_name, range_[0], range_[1], step=range_[2])
            elif distribution == "int":
                params_selection[param_name] = trial.suggest_int(param_name, range_[0], range_[1], step=range_[2])
            elif distribution == "power_int":
                power = trial.suggest_int(param_name, range_[0], range_[1], step=range_[2])
                params_selection[param_name] = range_[3]*(range_[4]**power)
            elif distribution == "multiply_int":
                multiplier = trial.suggest_int(param_name, range_[0], range_[1], step=range_[2])
                params_selection[param_name] = range_[3]*multiplier
            params_selection_str += f"{param_name}-{params_selection[param_name]}, "
        else:
            params_selection[param_name] = description

    return params_selection, params_selection_str

def objective(trial, optuna_params, model_params, training_params, x, y, model, training, save_model, device):

    # Instantiate Parameters
    params_selection_str = f"Starting Trial {trial.number}: "
    params_tr, params_selection_str = build_params(training_params, params_selection_str, trial)
    params_mo, params_selection_str = build_params(model_params, params_selection_str, trial)

    # Parameters selection log
    print(params_selection_str)
    logger.info(params_selection_str)




    stratifier = StratifiedKFold(n_splits=4, random_state=3, shuffle=True)

    sampling = optuna_params["sample"]
    if sampling is not None:
        ix = get_random_indices(y, sampling)
        x = x[ix]
        y = y[ix]

    metrics = ClassifierMetrics(optuna_params["metrics"])
    metric_fold = {metric: [] for metric in optuna_params["metrics"]}
    train_fold = []
    val_fold = []
    fold = 0
    for train_index, test_index in stratifier.split(x, y):

        model_instance = model(**params_mo)
        torch.compile(model_instance)
        params_tr["model"] = model_instance
        training_instance = training(**params_tr, device=device)

        x_train = x[train_index]
        y_train = y[train_index]
        x_test = x[test_index]
        y_test = y[test_index]
        train_loss = 1e10
        val_loss = 1e10

        # If pruning algorithm
        if fold == 0 and optuna_params["pruning"] is not None:
            for step in range(params_tr["n_iter"]):
                train_loss, val_loss = training_instance.train_step(x_train, y_train)
                trial.report(val_loss, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        else:
            train_loss, val_loss = training_instance.train_(x_train, y_train)
        train_fold.append(train_loss)
        val_fold.append(val_loss)

        preds = training_instance.predict_proba(x_test)
        scores = metrics.score_proba(y_test, preds)
        for name, score in scores.items():
            metric_fold[name].append(score)

        fold += 1

    # Store loss
    for name, metric in metric_fold.items():
        if name != "class_accuracy":
            print(f"{name}: {np.mean(metric):.3f} +/- {np.std(metric):.3f}")
        else:
            str_ = f"{name}: "
            for label in metric[0].keys():
                label_values = [value[label] for value in metric]
                str_ += f"{label}: {np.mean(label_values):.1f} +/-  {np.std(label_values):.1f} |"
            print(str_)

    optimization_value = np.mean(metric_fold[optuna_params["optimization"]])

    global best_val
    if best_val is None or optimization_value < best_val:
        best_val = optimization_value
        model_instance.save(save_model)

    return optimization_value


def optuna_optimization(optuna_params, model, training, X, y, model_params, training_params, save_dir, study_name,
                        device):

    save_pickle_params(model_params, save_dir / "model_params.json")
    save_pickle_params(training_params, save_dir / "training_params.json")

    # Create a study object and specify the optimization direction as 'minimize'.
    sampler = optuna.samplers.TPESampler(multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=50)
    study = optuna.create_study(study_name=study_name, direction="maximize", storage=f"sqlite:///{save_dir}/trial.db",
                                sampler=sampler)

    logger.info(f"Starting study with: {study_name} - stored in {save_dir}/trial.db")

    # Start the optimization; the number of trials can be adjusted
    objective_with_params = partial(objective, model_params=model_params, training_params=training_params,
                                    x=X, y=y, model=model, training=training, save_model=save_dir,
                                    optuna_params=optuna_params, device=device)
    study.optimize(objective_with_params, n_trials=50)

    # Print the optimal parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # Optionally, you can print the best value (e.g., lowest validation loss)
    best_value = study.best_value
    print(f"Best value (highest {optuna_params['optimization']}): {best_value}")