import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from collections import defaultdict

def plot_energy_accuracy_loss(log_file_path, individual=False):

    df = pd.read_csv(log_file_path)
    info_processed = {}

    for subj_id in list(set(df['subject_id'])):
        info_processed[int(subj_id)] = {
            'energy' : [],
            'accuracy': [],
            'loss': []
        }

        for step in list(set(df['adaptation_step'])):
            cur_df = df[(df['subject_id'] == subj_id) & (df['adaptation_step'] == step)]
            info_processed[subj_id]['energy'].append(cur_df['energy'].mean())
            info_processed[subj_id]['loss'].append(cur_df['loss'].mean())
            info_processed[subj_id]['accuracy'].append(cur_df['accuracy'].mean())

    # plot
    minmax_norm = lambda data: (data - data.min()) / (data.max() - data.min())

    all_subj_id = list(set(df['subject_id']))

    if individual:
        rows, cols = 2, 4
    else:
        rows, cols = 3, 3

    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))

    for i in range(rows):
        for j in range(cols):
            subj_id = all_subj_id[i * cols + j] if individual else i * rows + j + 1
            print(subj_id)
            ax = axes[i, j]
            ax.plot(
                [i for i in range(1, len(info_processed[subj_id]['energy']) + 1)],
                minmax_norm(np.array(info_processed[subj_id]['energy'])),
                'o-',
                label = 'Energy',
                color = 'C0',
                linewidth = 2
            )
            ax.set_xlabel('Adaptation Steps')
            ax.set_xticks([i for i in range(1, len(info_processed[subj_id]['accuracy']) + 1)])
            ax.set_ylabel('Normalized Energy Score', color='C0')
            ax.tick_params(axis='y', labelcolor='C0')

            ax.plot(
                [i for i in range(1, len(info_processed[subj_id]['loss']) + 1)],
                minmax_norm(np.array(info_processed[subj_id]['loss'])),
                '^--',
                label='Loss',
                color='gray',
                linewidth=1.5
            )

            ax2 = ax.twinx()
            #ax3.spines['right'].set_position(('axes', 1.1))  # Offset to right
            ax2.set_ylim(min(info_processed[subj_id]['loss']),
                         max(info_processed[subj_id]['loss']))
            ax2.tick_params(axis='y', labelcolor='gray')

            if max(info_processed[subj_id]['accuracy']) != min(info_processed[subj_id]['accuracy']):
                ax.plot(
                    [i for i in range(1, len(info_processed[subj_id]['accuracy']) + 1)],
                    minmax_norm(np.array(info_processed[subj_id]['accuracy'])),
                    'x--',
                    label='Accuracy',
                    color='brown',
                    linewidth=2
                )

                ax3 = ax.twinx()
                ax3.spines['right'].set_position(('axes', 1.1))
                ax3.set_ylim(min(info_processed[subj_id]['accuracy']),
                             max(info_processed[subj_id]['accuracy']))
                ax3.tick_params(axis='y', labelcolor='maroon')

                ax.text(1.1, -0.05, "Acc", transform=ax.transAxes,
                        rotation=0, ha='left', va='top', color='maroon', fontsize=10)

            energy_diff = max(info_processed[subj_id]['energy']) - min(info_processed[subj_id]['energy'])
            ax.set_title(f'Subject ID : {subj_id}, ΔEnergy: {energy_diff:.5f}')
            ax.text(1, -0.05, "Loss", transform=ax.transAxes,
                    rotation=0, ha='left', va='top', color='gray', fontsize=10)
            ax.grid(True)



    plt.tight_layout()
    plt.show()


def plot_accuracy(acc_list, configs):

    all_subj_ids = sorted({subj_id for acc in acc_list for subj_id in acc})
    x = np.arange(len(all_subj_ids))  # the label locations
    width = 0.8 / len(configs)  # width of the bars (divided for group)

    plt.figure(figsize=(10, 6))

    for i, (acc, config) in enumerate(zip(acc_list, configs)):
        acc_vals = []
        for subj_id in all_subj_ids:
            val = acc.get(subj_id, np.nan)
            if isinstance(val, dict):
                acc_vals.append(val.get('test_acc', np.nan))
            else:
                acc_vals.append(val)

        print(f'avg acc {config} : {np.nanmean(acc_vals)}')
        plt.bar(x + i * width, acc_vals, width, label=config)

    plt.xlabel('Subj id')
    plt.ylabel('Accuracy')
    plt.xticks(x + width * (len(configs) - 1) / 2, all_subj_ids, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_energy_per_batch(log_file_path):

    df = pd.read_csv(log_file_path)
    info_processed = {}

    for subj_id in list(set(df['subject_id'])):
        mean_energy = []
        info_processed[subj_id] = defaultdict(int)
        for batch in list(set(df[df['subject_id'] == subj_id]['batch'])):
            cur_df = df[(df['subject_id'] == subj_id) & (df['batch'] == batch)]
            info_processed[subj_id][batch] = list(cur_df['energy'])
        for step in list(set(df['adaptation_step'])):
            cur_df = df[(df['subject_id'] == subj_id) & (df['adaptation_step'] == step)]
            mean_energy.append(cur_df['energy'].mean())
        info_processed[subj_id]['mean_energy'] = mean_energy

    fig, axes = plt.subplots(3, 3, figsize=(20, 10))
    for i in range(3):
        for j in range(3):
            subj_id = i*3 + j + 1
            ax = axes[i, j]
            for batch, energy_list in info_processed[subj_id].items():
                if batch == 'mean_energy':
                    ax.plot(
                        [i for i in range(1, len(energy_list) + 1)],
                        energy_list,
                        'x--',
                        label='Mean Energy',
                        linewidth=2
                    )
                else:
                    ax.plot(
                        [i for i in range(1, len(energy_list) + 1)],
                        energy_list,
                        label=f'Batch {batch}',
                        linewidth=2
                    )
            ax.set_xlabel('Adaptation Steps')
            ax.legend()
            ax.set_ylabel('Energy Score')
            ax.set_xlabel('Adaptation steps')
            ax.set_title(f'subject id : {subj_id}')
            ax.grid()
    plt.tight_layout()
    plt.show()

log_path = 'adaptation_data.csv'
plot_energy_per_batch(log_path)
plot_energy_accuracy_loss(log_path)
# filepath_lst = [
#     '/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logs/src-bcic2a_loso_2023-12-04_14-41-13_no_adaptation_accuracy.json',
#     '/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logs/src-bcic2a_loso_2023-12-04_14-41-13_entropy_minimization_accuracy.json',
# '/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logs/src-bcic2a_loso_2023-12-04_14-41-13_energy_adaptation_only_batch_accuracy.json',
# '/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logs/src-bcic2a_loso_2023-12-04_14-41-13_energy_adaptation_adapt_batch_only.json']
# configs = ['source', 'entropy minimization', 'energy', 'energy adapt ber patch']
# acc_list = []
#
# for filepath in filepath_lst:
#     with open(filepath, 'r') as f:
#         acc_list.append(json.load(f))
#
# plot_accuracy(acc_list, configs)