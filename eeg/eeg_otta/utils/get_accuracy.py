from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

def corrupt(x, severity=5):

    stds = [8, 12, 18, 26, 38]
    std = stds[severity - 1]

    noise = torch.randn_like(x).to(x.device) * std
    corrupted = x + noise
    return corrupted


def get_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device):
    outputs, labels = [], []
    with torch.no_grad():
        for step in range(model.hyperparams['adaptation_steps']):
            model.step += 1
            model.adapt=True
            for batch in tqdm(data_loader):
                x, y = batch
                output = torch.softmax(model(x.to(device), y), -1)

            model.adapt = False
            model.batch = 0
            for batch in tqdm(data_loader):
                x, y = batch
                output = torch.softmax(model(x.to(device), y), -1)

                if step == model.hyperparams['adaptation_steps'] - 1:
                    outputs.append(output)
                    labels.append(y)
                model.batch += 1

    outputs = torch.concatenate(outputs)
    labels = torch.concatenate(labels)

    y_pred = outputs.argmax(-1).cpu()
    accuracy = (y_pred == labels).float().numpy().mean()

    return accuracy
