from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader


def get_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device):
    def adapt():
        model.adapt = True
        for batch in tqdm(data_loader):
            x, y = batch
            output = torch.softmax(model(x.to(device), y), -1)
    
    def predict():
        outputs, labels = [], []
        model.adapt = False
        model.batch = 0
        for batch in tqdm(data_loader):
            x, y = batch
            output = torch.softmax(model(x.to(device), y), -1)
            outputs.append(output)
            labels.append(y)
            # model.batch += 1
        
        outputs = torch.concatenate(outputs)
        labels = torch.concatenate(labels)
        
        y_pred = outputs.argmax(-1).cpu()
        accuracy = (y_pred == labels).float().numpy().mean()
        return accuracy
    
    with torch.no_grad():
        accuracy = predict()
        print(f"accuracy: {100 * accuracy:.2f}%")
        
        for step in range(model.config.get("hyperparams", {}).get('adaptation_steps', 0)):
            # model.step += 1
            adapt()
            accuracy = predict()
            print(f"accuracy: {100 * accuracy:.2f}%")
            
    return accuracy
