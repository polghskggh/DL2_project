from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device):
    outputs, labels = [], []
    subject_ids = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            x, y = batch
            y, s_ids = y.T
            output = torch.softmax(model(x.to(device)), -1)
            outputs.append(output)
            labels.append(y)
            subject_ids.append(s_ids)

    outputs = torch.concatenate(outputs)
    labels = torch.concatenate(labels)
    subject_ids = torch.concatenate(subject_ids)

    y_pred = outputs.argmax(-1).cpu()
    correct = (y_pred == labels).float().cpu()
    accs = {}
    for i in torch.unique(subject_ids):
        accs[f"subject_{i}"] = correct[subject_ids == i].mean().item()
    accs["total"] = correct.mean().item()

    return accs
