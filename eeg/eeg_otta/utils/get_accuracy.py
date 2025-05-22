from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

def corrupt(x, severity=5):

    stds = [0.1, 0.2, 3, 10, 40]
    std = stds[severity - 1]

    noise = torch.randn_like(x).to(x.device) * std
    corrupted = x + noise
    return corrupted

def get_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device):
    outputs, labels = [], []
    with torch.no_grad():
        for step in range(getattr(model, 'hyperparams', {}).get('adaptation_steps', 5)):
            if isinstance(getattr(model, 'step', 'false'), int):
                model.step += 1
            model.adapt = True
            for batch in tqdm(data_loader):
                x, y = batch
                x = corrupt(x, severity=5)
                output = torch.softmax(model(x.to(device), y), -1)

            model.adapt = False
            if isinstance(getattr(model, 'batch', 'false'), int):
                model.batch = 0
            for batch in tqdm(data_loader):
                x, y = batch
                x = corrupt(x, severity=5)
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

    return accuracy
