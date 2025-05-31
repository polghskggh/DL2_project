from tqdm import tqdm
from pathlib import Path

import os
import numpy as np
from copy import deepcopy
import json

import torch
from torch import nn
from torch.utils.data import DataLoader
from .seed import seed_everything

CHECKPOINT_PATH = os.path.join(Path(__file__).resolve().parents[2], "checkpoints")

def corrupt(x, severity=5):

    stds = [0.1, 1, 10, 20, 40]
    std = stds[severity - 1]

    noise = torch.randn_like(x).to(x.device) * std
    corrupted = x + noise
    return corrupted

def get_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device):
    outputs, labels = [], []
    with torch.no_grad():
        for step in range(getattr(model, 'hyperparams', {}).get('adaptation_steps', 1)):
            if isinstance(getattr(model, 'step', 'false'), int):
                model.step += 1
            model.adapt = True
            for batch in tqdm(data_loader):
                x, y = batch
                #x = corrupt(x, severity=5)
                output = torch.softmax(model(x.to(device), y), -1)

            model.adapt = False
            if isinstance(getattr(model, 'batch', 'false'), int):
                model.batch = 0
            for batch in data_loader:
                x, y = batch
                #x = corrupt(x, severity=5)
                output = torch.softmax(model(x.to(device), y), -1)

                if step == getattr(model, 'hyperparams', {}).get('adaptation_steps', 1) - 1:
                    outputs.append(output)
                    labels.append(y)
                if isinstance(getattr(model, 'batch', 'false'), int):
                    model.batch += 1

    outputs = torch.concatenate(outputs)
    labels = torch.concatenate(labels)

    y_pred = outputs.argmax(-1).cpu()
    accuracy = (y_pred == labels).float().numpy().mean()

    return accuracy, model


def calculate_accuracy(model_cls, tta_cls, datamodule, config, get_model_dict=False):
    print(f"starting run for seed: {config['seed']}")
    model_dict = {}
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
                                 "model.ckpt")
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
            cal_acc, _ = get_accuracy(model, datamodule.calibration_dataloader(), device)
            cal_accs.append(cal_acc)

            print(f"cal_acc subject {subject_id}: {100 * cal_accs[-1]:.2f}%")

        acc, model = get_accuracy(model, datamodule.test_dataloader(), device)
        if get_model_dict:
            print('updating model dict')
            model_dict[subject_id] = deepcopy(model)

        test_accs.append(acc)
        test_acc_logs[subject_id] = float(acc.item())
        print(f" test_acc subject {subject_id}: {100 * test_accs[-1]:.2f}%")

    print(f"test_acc: {100 * np.mean(test_accs):.2f}")

    with open(os.path.join(f"{config['tta_config']['save_dir']}", f"accuracy_{config['seed']}.json"), 'w') as f:
        json.dump(test_acc_logs, f)

    return test_accs, model_dict