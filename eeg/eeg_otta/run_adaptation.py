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

def run_adaptation(config):
    # load source config
    with open(os.path.join(CHECKPOINT_PATH, config["source_run"], "config.yaml")) as f:
        source_config = yaml.safe_load(f)
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")

    datamodule_cls = get_datamodule_cls(source_config["dataset_name"])
    model_cls = BaseNet
    tta_cls = get_tta_cls(config["tta_method"])

    if source_config["subject_ids"] == "all":
        subject_ids = datamodule_cls.all_subject_ids
    else:
        subject_ids = [source_config["subject_ids"]]

    source_config["preprocessing"]["alignment"] = False

    if config["online"]:
        source_config["preprocessing"]["batch_size"] = 1
    datamodule = datamodule_cls(source_config["preprocessing"], subject_ids=subject_ids)

    cal_accs, test_accs = [], []
    test_acc_logs = {}
    for version, subject_id in enumerate(subject_ids):
        seed_everything(source_config["seed"])

        # load checkpoint
        ckpt_path = os.path.join(CHECKPOINT_PATH, config["source_run"], str(subject_id),
                                 "model-v1.ckpt")
        model = model_cls.load_from_checkpoint(ckpt_path, map_location=device)

        # set subject_id
        datamodule.subject_id = subject_id
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

    # print overall test accuracy
    if config.get("continual", False):
        print(f"cal_acc: {100 * np.mean(cal_accs):.2f}")
    print(f"test_acc: {100 * np.mean(test_accs):.2f}")

    with open(f'./logs/{config["source_run"]}_{config["tta_method"]}_accuracy.json', 'w') as f:
        json.dump(test_acc_logs, f)

    return np.mean(test_accs)

def tune(config):
    def objective(trial):
        hyperparams = {
            'sgld_steps': trial.suggest_int("sgld_steps", 5, 200),
            'sgld_lr': trial.suggest_float("sgld_lr", 1e-5, 1, log=True),
            'sgld_std': trial.suggest_float("sgld_std", 1e-5, 1),
            'reinit_freq': trial.suggest_float("reinit_freq", 1e-5, 1),
        }
        config['hyperparams'] = hyperparams
        return run_adaptation(config)

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
    parser.add_argument("--online", default=True, action="store_true")
    parser.add_argument("--tune", default=True, action="store_true")
    args = parser.parse_args()

    # load config
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)

    config["online"] = args.online
    if args.tune:
        tune(config)
    else:
        run_adaptation(config)
