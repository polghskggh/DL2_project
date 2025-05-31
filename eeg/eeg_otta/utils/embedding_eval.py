import torch
from tqdm import tqdm
import numpy as np

from eeg_otta.utils.get_accuracy import corrupt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from matplotlib import pyplot as plt

def forward(tea_model, x):
    x = x.to(tea_model.device)
    logits, embeds = tea_model.energy_model.f(x, return_embeds=True)
    energy = -torch.logsumexp(logits, dim=-1)
    return energy, logits, embeds

def prepare_labeled_data(tea_model, data_loader, do_corrupt=False):
    aux = {
        'logits': [],
        'energies': [],
        'embeds': [],
        'predictions': []
    }
    tea_model.energy_model.eval()
    with torch.no_grad():
        for sample in tqdm(data_loader):
            x, y = sample
            if do_corrupt:
                x = corrupt(x, 0)

            energy, output, embed = forward(tea_model, x)
            prediction = torch.argmax(torch.softmax(output, -1), -1)

            for storage, datum in zip(aux.values(), [output, energy, embed, y, prediction]):
                storage.append(datum)

    for key, value in aux.items():
        aux[key] = torch.cat(value, dim=0).detach().cpu().numpy()
    return aux

def reduce_embedding(model, train_data, test_data, method='pca'):
    train_features = prepare_labeled_data(model, train_data)
    test_features = prepare_labeled_data(model, test_data)
    num_train = train_features['embeds'].shape[0]

    embeddings = np.concatenate((train_features['embeds'], test_features['embeds']), axis=0)
    labels = np.concatenate((train_features['predictions'], test_features['predictions']), axis=0)

    if method == 'pca':
        reduced = PCA(n_components=2).fit_transform(embeddings)
    elif method == 'tsne':
        reduced = TSNE(n_components=2, random_state=0).fit_transform(embeddings)
    else:
        raise ValueError('Unknown embedding method: {}'.format(method))
    distribution = np.zeros(embeddings.shape[0])
    distribution[:num_train] = 1
    dataset = {
        'embeddings': reduced,
        'distribution': distribution,
        'labels': labels,
    }
    return dataset

def plot_embeddings(model, train_data, test_data, method='pca'):
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['x', 'o']
    dataset = reduce_embedding(model, train_data, test_data, method=method)
    for split in np.unique(dataset['distribution']):
        idxs = dataset['distribution'] == split
        embeddings = dataset['embeddings'][idxs]
        labels = dataset['labels'][idxs]
        labels = [colors[label] for label in labels]
        plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, marker=markers[int(split)], alpha=0.2)

    plt.title(f"{method} visualization of the last embedding")
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
    plt.grid(True)
    plt.show()
