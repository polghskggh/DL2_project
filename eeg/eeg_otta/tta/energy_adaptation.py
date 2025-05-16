import csv
from mne.filter import filter_data

import mne
import torch.nn as nn
import torch.jit
import torch
from copy import deepcopy
import numpy as np
from mne.filter import filter_data

from .alignment import OnlineAlignment
from .base import TTAMethod


class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):
        penult_z = self.f(x)
        return penult_z

    def forward(self, x, y=None):
        logits = self.classify(x)
        if y is None:
            return -logits.logsumexp(1), logits
        else:
            return -torch.gather(logits, 1, y[:, None]), logits


class EnergyAdaptation(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(EnergyAdaptation, self).__init__(model, config, info)
        self.energy_model = EnergyModel(model)
        self.replay_buffer = []

        self.hyperparams = config['hyperparams']
        self.subject_id = config['subject_id']
        self.batch = 0
        self.csv_file = f'./logs/{config["log_name"]}.csv'
        self.adapt= True
        header = ['subject_id', 'batch', 'adaptation_step', 'loss', 'energy', 'accuracy']
        if config['initialise_log']:
            with open(self.csv_file, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(header)


    def forward_sliding_window(self, x):
        if self.config.get("alignment", False):
            # align data
            x = OnlineAlignment.align_data(
                x, self.config.get("alignment"),
                self.config.get("averaging_method", "equal"),
                self.config.get("align_alpha", None))
        outputs = self.model(x)
        return outputs

    @staticmethod
    def init_random(bs, series_length=1000, n_channels=22):
        return torch.normal(0, 0.0363, size=(bs, n_channels, series_length))

    @staticmethod
    def generate_pink_noise(bs: int, series_length: int, n_channels: int, alpha: float = 1) -> torch.Tensor:
        pink = np.zeros((bs, n_channels, series_length))
        for i in range(bs):
            for c in range(n_channels):
                white = np.random.randn(series_length)
                f = np.fft.rfftfreq(series_length)
                f[0] = 1  # avoid divide-by-zero
                spectrum = np.fft.rfft(white) / (f ** (alpha / 2))
                pink_signal = np.fft.irfft(spectrum, n=series_length)
                pink[i, c, :] = pink_signal / np.std(pink_signal) * 0.0363
        return torch.tensor(pink, dtype=torch.float32)

    @staticmethod
    def _sample_p_0(reinit_freq, replay_buffer, bs, series_length, n_channels, device, y=None, train_dataset=None, noise_alpha=1):
        if train_dataset is not None:
            inds = np.random.randint(0, len(train_dataset), bs)
            #print(train_dataset[inds][0].shape)
            samples = train_dataset[inds][0]
            noise = EnergyAdaptation.generate_pink_noise(bs, series_length, n_channels, alpha=noise_alpha)
            samples += 0.2 * noise
            #print(samples.shape)
            return samples.to(device), inds

        if len(replay_buffer) == 0:
            return EnergyAdaptation.generate_pink_noise(bs, series_length=series_length, n_channels=n_channels, alpha=noise_alpha), []
        buffer_size = len(replay_buffer)
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds

        buffer_samples = replay_buffer[inds]
        random_samples = EnergyAdaptation.generate_pink_noise(bs, series_length=series_length, n_channels=n_channels, alpha=noise_alpha)
        choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(self, sgld_steps, sgld_lr, sgld_std, reinit_freq, adaptation_steps,
                 batch_size, series_length, n_channels, device, y=None, train_dataset=None, apply_filter=False, align=False, noise_alpha=1):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        # self.energy_model.eval()
        # get batch size
        #print(sgld_steps)
        bs = batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = self._sample_p_0(
            reinit_freq=reinit_freq,
            replay_buffer=self.replay_buffer,
            bs=bs,
            series_length=series_length,
            n_channels=n_channels,
            device=device,
            y=y,
            train_dataset=train_dataset,
            noise_alpha=noise_alpha,
        )
        init_samples = deepcopy(init_sample)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True).to(self.device)
        # sgld

        for k in range(sgld_steps):
            f_prime = torch.autograd.grad(self.energy_model(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
            x_k.data -= sgld_lr * f_prime + sgld_std * self.generate_pink_noise(bs, series_length, n_channels, alpha=noise_alpha).to(self.device)                
        # self.energy_model.train()
        final_samples = x_k.detach()

        preprocess_config = self.config.get("preprocessing")
        if apply_filter and preprocess_config is not None:
            l_freq, h_freq = preprocess_config["low_cut"], preprocess_config["high_cut"]

            if l_freq is not None or h_freq is not None:
                final_samples = filter_data(final_samples.cpu().numpy().astype(np.float64()), sfreq=preprocess_config["sfreq"], l_freq=l_freq, h_freq=h_freq, verbose=False)
                final_samples = torch.tensor(final_samples, dtype=torch.float32).to(device)
        
        if align and preprocess_config is not None:
            final_samples = OnlineAlignment.align_data(
                final_samples, method=preprocess_config.get("alignment")
            )

        # update replay buffer

        if train_dataset is None and len(self.replay_buffer) > 0:
            self.replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples, init_samples.detach()

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, y=None, train_dataset=None):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """

        if self.config.get("alignment", False):
            # align data
            x = OnlineAlignment.align_data(
                x, self.config.get("alignment"),
                self.config.get("averaging_method", "equal"),
                self.config.get("align_alpha", None))

        batch_size = x.shape[0]
        n_channels = x.shape[1]
        series_length = x.shape[2]
        device = x.device
        logs = []

        for step in range(self.hyperparams['adaptation_steps']):
            x_fake, _ = self.sample_q(**self.hyperparams, batch_size=batch_size, series_length=series_length,
                                      n_channels=n_channels, device=device, y=None, train_dataset=train_dataset)

            # forward
            out_real = self.energy_model(x)
            energy_real = out_real[0].mean()
            energy_fake = self.energy_model(x_fake)[0].mean()

            # adapt
            self.optimizer.zero_grad()

            alpha = self.hyperparams['energy_real_weight']
            loss = alpha * energy_real - (1 - alpha) * energy_fake

            loss.backward()
            self.optimizer.step()

            if y is not None:
                outputs = self.energy_model.classify(x)
                accuracy = (outputs.argmax(-1).cpu() == y).float().numpy().mean()
                logs.append((self.subject_id, self.batch, step, loss.item(), energy_real.item(), accuracy))

        outputs = self.energy_model.classify(x)
        if len(logs) > 0:
            with open(self.csv_file, mode='a') as file:
                writer = csv.writer(file)
                writer.writerows(logs)
        self.batch += 1
        return outputs

    def configure_model(self):
        self.model.eval()  # eval mode to avoid using dropout during test-time
        # self.model.requires_grad_(True)
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            else:
                m.requires_grad_(False)

    def forward(self, x, y, train_dataset=None):
        if self.adapt:
            return self.forward_and_adapt(x, y, train_dataset=train_dataset)
        else:
            print('no adapt')
            return self.forward_sliding_window(x)
