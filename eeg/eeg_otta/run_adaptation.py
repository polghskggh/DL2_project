from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import yaml

from utils.get_accuracy import calculate_accuracy
from utils.config_setup import load_config, setup

CHECKPOINT_PATH = os.path.join(Path(__file__).resolve().parents[1], "checkpoints")
CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "tta_energy_2a.yaml"

import optuna

def run_adaptation(config):
    config = load_config(config)

    os.makedirs(f"./logs/{config['source_run']}", exist_ok=True)

    hyperparams = {
        'sgld_steps': 1,
        'sgld_lr': 0.017975217955105576,
        'sgld_std': 0.01115193577977435,
        'reinit_freq': 0.05,
        'adaptation_steps': 1,
        'energy_real_weight': 0.5172643563168701,
        'apply_filter': True,
        'align': False,
        'noise_alpha': 1.050088336278543,
    }

    config['tta_config']['hyperparams'] = hyperparams
    model_cls, tta_cls, datamodule = setup(config)
    corruption_levels = [None]

    for corruption_level in corruption_levels:
        datamodule.corruption_level = corruption_level
        save_dir = f"./logs/{config['source_run']}/{config['tta_config']['log_name']}_{corruption_level}"
        os.makedirs(save_dir, exist_ok=True)
        config['tta_config']['save_dir'] = save_dir

        for seed in range(0,3):
            config['seed'] = seed
            config['tta_config']['seed'] = seed
            test_accs, _ = calculate_accuracy(model_cls, tta_cls, datamodule, config)

def tune(config, n_trials=1):
    def objective(trial):
        hyperparams = {
            'sgld_steps': trial.suggest_int('sgld_steps', 10, 30),
            'sgld_lr': trial.suggest_float('sgld_lr', 1e-2, 3, log=True),
            'sgld_std': trial.suggest_float('sgld_std', 1e-3, 1e-1, log=True),
            'reinit_freq': 0.05,
            'adaptation_steps': trial.suggest_int('adaptation_steps', 1, 8),
            'energy_real_weight': trial.suggest_float('energy_real_weight', 1e-1, 1),
            'apply_filter': True,
            'align':False,
            'noise_alpha': trial.suggest_float('noise_alpha', 0.0, 1.5),
        }
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 288])
        
        # optimizer = trial.suggest_categorical("optimizer", ["SGD", "Adam"])
        optimizer_kwargs = {
            'lr': trial.suggest_float('lr', 1e-4, 3e-2, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            # 'momentum': trial.suggest_float('momentum', 0.8, 0.99),
        }

        config_local = load_config(config)
        config_local["preprocessing"]["batch_size"] = batch_size
        config_local['tta_config']['hyperparams'] = hyperparams
        # config_local['tta_config']['optimizer'] = optimizer
        config_local['tta_config']['optimizer_kwargs'] = optimizer_kwargs
        model_cls, tta_cls, datamodule = setup(config_local)
        test_accs, _ = calculate_accuracy(model_cls, tta_cls, datamodule, config_local)
        return np.mean(test_accs)

    # Define the local SQLite database file
    storage_url = "sqlite:///tea.db"
    study_name = "tea_eeg"
    # Create or load an existing study
    study = optuna.create_study(storage=storage_url, study_name=study_name, direction="maximize",
                                load_if_exists=True)

    study.optimize(objective, n_trials=1000)


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--online", default=False, action="store_true")
    parser.add_argument("--tune", default=False, action="store_true")
    parser.add_argument("--trials", default=1, type=int)
    args = parser.parse_args()

    # load config
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)

    config["online"] = args.online
    if args.tune:
        tune(config, n_trials=args.trials)
    else:
        run_adaptation(config)
