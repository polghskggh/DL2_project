import copy
from argparse import ArgumentParser
import os
from pathlib import Path

import numpy as np
import torch
import yaml
import json

from models import BaseNet
from utils.get_accuracy import get_accuracy
from utils.get_datamodule_cls import get_datamodule_cls
from utils.get_tta_cls import get_tta_cls
from utils.seed import seed_everything

CHECKPOINT_PATH = os.path.join(Path(__file__).resolve().parents[1], "checkpoints")
CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "tta_energy.yaml"

import optuna

def load_config(config):
    with open(os.path.join(CHECKPOINT_PATH, config["source_run"], "config.yaml")) as f:
        source_config = yaml.safe_load(f)

    for key, value in config.items():
        source_config[key] = value
    return source_config

def setup(config):
    # load source config
    datamodule_cls = get_datamodule_cls(config["dataset_name"])
    model_cls = BaseNet
    tta_cls = get_tta_cls(config["tta_method"])

    if config["subject_ids"] == "all":
        subject_ids = datamodule_cls.all_subject_ids
    elif isinstance(config["subject_ids"], int):
        subject_ids = [config["subject_ids"]]
    else:
        subject_ids = config["subject_ids"]

    config["subject_ids"] = subject_ids

    if config["tta_config"]["alignment"]:
        config["preprocessing"]["alignment"] = False

    datamodule = datamodule_cls(config["preprocessing"], subject_ids=subject_ids)
    return model_cls, tta_cls, datamodule

def calculate_accuracy(model_cls, tta_cls, datamodule, config):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")

    cal_accs, test_accs = [], []
    test_acc_logs = {}

    config['train_individual'] = config.get("train_individual", False)

    subject_id_lst = [_id for _id in datamodule.all_subject_ids if _id not in config["subject_ids"]] \
        if config['train_individual'] \
        else config["subject_ids"]

    for version, subject_id in enumerate(subject_id_lst):
        seed_everything(config["seed"])

        # load checkpoint
        ckpt_path = os.path.join(CHECKPOINT_PATH, config["source_run"],
                                 str(config["subject_ids"][0]) if config['train_individual'] else str(subject_id),
                                 "model.ckpt" if config['train_individual'] else "model-v1.ckpt")
        model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)

        # set subject_id
        datamodule.subject_id = subject_id
        config['tta_config']['subject_id'] = subject_id
        config['tta_config']['preprocessing'] = config['preprocessing']
        if config['train_individual']:
            config['tta_config']['initialise_log'] = version == 0
        else:
            config['tta_config']['initialise_log'] = subject_id == 1
        
        datamodule.prepare_data()
        datamodule.setup()

        model = tta_cls(model, config["tta_config"], datamodule.info)

        if config.get("continual", False):
            cal_acc = get_accuracy(model, datamodule.calibration_dataloader(), device)
            cal_accs.append(cal_acc)
            print(f"cal_acc subject {subject_id}: {100 * cal_accs[-1]:.2f}%")

        acc = get_accuracy(model, datamodule.test_dataloader(), device)
        test_accs.append(acc)
        test_acc_logs[subject_id] = float(acc.item())
        print(f" test_acc subject {subject_id}: {100 *test_accs[-1]:.2f}%")
    return test_accs, test_acc_logs

def run_adaptation(config):
    config = load_config(config)

    hyperparams = {
        'sgld_steps': 26,
        'sgld_lr': 0.03098715690500288,
        'sgld_std': 0.0033756671301297617,
        'reinit_freq': 0.05,
        'adaptation_steps': 3,
        'energy_real_weight': 0.7,
        'apply_filter': True,
        'align': False,
        'noise_alpha': 1.1021171479575294,
    }

    config['tta_config']['hyperparams'] = hyperparams
    model_cls, tta_cls, datamodule = setup(config)
    test_accs, test_acc_logs = calculate_accuracy(model_cls, tta_cls, datamodule, config)
    # print overall test accuracy
    print(f"test_acc: {100 * np.mean(test_accs):.2f}")

    with open(f'./logs/{config["source_run"]}_{config["tta_config"]["log_name"]}_accuracy.json', 'w') as f:
        json.dump(test_acc_logs, f)

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
        config_local['tta_config']['optimizer_kwargs'] |= optimizer_kwargs
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
