from functools import partial

import optuna


def objective(trial, model_params, training_params, x, y, model, training):

    # Instantiate Parameters
    params_selection_str = f"Starting Trial {trial.number}: "
    params_tr = {}
    for param_name, description in training_params.items():
        if len(description) > 1:
            range_ = description[0]
            distribution = description[1]
            if distribution == "log_uniform":
                params_tr[param_name] = trial.suggest_loguniform(param_name, range_[0], range_[1])
            elif distribution == "uniform":
                params_tr[param_name] = trial.suggest_uniform(param_name, range_[0], range_[1])
            elif distribution == "categorical":
                params_tr[param_name] = trial.suggest_categorical(param_name, range_)
            params_selection_str += f"{param_name}-{params_tr[param_name]}, "

        else:
            params_tr[param_name] = description[0]
    params_model = {}
    for param_name, description in model_params.items():
        if isinstance(description, list):
            range_ = description[0]
            distribution = description[1]
            if distribution == "log_uniform":
                params_model[param_name] = trial.suggest_loguniform(param_name, range_[0], range_[1])
            elif distribution == "uniform":
                params_model[param_name] = trial.suggest_uniform(param_name, range_[0], range_[1])
            elif distribution == "categorical":
                params_model[param_name] = trial.suggest_categorical(param_name, range_)
            params_selection_str += f"{param_name}-{params_model[param_name]}, "
        else:
            params_model[param_name] = description

    print(params_selection_str)

    # Create Instances
    model_instance = model(**model_params)
    params_tr["model"] = model_instance
    training_instance = training(**params_tr)

    return training_instance.train_(x, y)


def optuna_optimization(model, training, X, y, model_params, training_params):

    # Create a study object and specify the optimization direction as 'minimize'.
    study = optuna.create_study(study_name="test_optimization", direction="minimize")

    # Start the optimization; the number of trials can be adjusted
    objective_with_params = partial(objective, model_params=model_params, training_params=training_params,
                                    x=X, y=y, model=model, training=training)
    study.optimize(objective_with_params, n_trials=2)

    # Print the optimal parameters
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # Optionally, you can print the best value (e.g., lowest validation loss)
    best_value = study.best_value
    print(f"Best value (lowest validation loss): {best_value}")