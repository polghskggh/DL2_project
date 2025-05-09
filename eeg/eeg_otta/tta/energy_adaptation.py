import csv

import mne
import torch.nn as nn
import torch.jit
import torch
from copy import deepcopy

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
        self.csv_file = f'logged_data.csv'
        header = ['subject_id', 'batch', 'adaptation_step', 'loss', 'energy', 'accuracy']
        if self.subject_id == 1:
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
    def _sample_p_0(reinit_freq, replay_buffer, bs, series_length, n_channels, device, y=None):
        if len(replay_buffer) == 0:
            return EnergyAdaptation.init_random(bs, series_length=series_length, n_channels=n_channels), []
        buffer_size = len(replay_buffer)
        inds = torch.randint(0, buffer_size, (bs,))
        # if cond, convert inds to class conditional inds

        buffer_samples = replay_buffer[inds]
        random_samples = EnergyAdaptation.init_random(bs, series_length=series_length, n_channels=n_channels)
        choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None, None]
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
        return samples.to(device), inds

    def sample_q(self, sgld_steps, sgld_lr, sgld_std, reinit_freq, adaptation_steps,
                 batch_size, series_length, n_channels, device, y=None):
        """this func takes in replay_buffer now so we have the option to sample from
        scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
        """
        # self.energy_model.eval()
        # get batch size
        bs = batch_size if y is None else y.size(0)
        # generate initial samples and buffer inds of those samples (if buffer is used)
        init_sample, buffer_inds = self._sample_p_0(reinit_freq=reinit_freq, replay_buffer=self.replay_buffer, bs=bs, series_length=series_length, n_channels=n_channels, device=device ,y=y)
        init_samples = deepcopy(init_sample)
        x_k = torch.autograd.Variable(init_sample, requires_grad=True).to(self.device)
        # sgld

        for k in range(sgld_steps):
            f_prime = torch.autograd.grad(self.energy_model(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
            x_k.data -= sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
        # self.energy_model.train()
        final_samples = x_k.detach()
        # update replay buffer
        if len(self.replay_buffer) > 0:
            self.replay_buffer[buffer_inds] = final_samples.cpu()
        return final_samples, init_samples.detach()

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x, y=None):
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
                                      n_channels=n_channels, device=device, y=None)

            # forward
            out_real = self.energy_model(x)
            energy_real = out_real[0].mean()
            energy_fake = self.energy_model(x_fake)[0].mean()

            # adapt
            self.optimizer.zero_grad()
            loss = energy_real - energy_fake
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

    def forward(self, x, y):
        return self.forward_and_adapt(x, y)
