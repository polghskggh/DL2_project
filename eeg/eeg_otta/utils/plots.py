import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

def plot_energy_accuracy_loss(log_file_path):

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
    fig, axes = plt.subplots(3, 3, figsize=(20, 10))

    for i in range(3):
        for j in range(3):
            subj_id = i*3 + j + 1
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

            energy_diff = max(info_processed[i*3 + j + 1]['energy']) - min(info_processed[i*3 + j + 1]['energy'])
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
plot_energy_accuracy_loss('/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logged_data_adapt_per_batch.csv')

filepath_lst = [
    '/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logs/src-bcic2a_loso_2023-12-04_14-41-13_no_adaptation_accuracy.json',
    '/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logs/src-bcic2a_loso_2023-12-04_14-41-13_entropy_minimization_accuracy.json',
'/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logs/src-bcic2a_loso_2023-12-04_14-41-13_energy_adaptation_only_batch_accuracy.json',
'/Users/tyme/Desktop/University/Block_5/FOMO/TEA/eeg/logs/src-bcic2a_loso_2023-12-04_14-41-13_energy_adaptation_adapt_batch_only.json']
configs = ['source', 'entropy minimization', 'energy', 'energy adapt ber patch']
acc_list = []

for filepath in filepath_lst:
    with open(filepath, 'r') as f:
        acc_list.append(json.load(f))

plot_accuracy(acc_list, configs)