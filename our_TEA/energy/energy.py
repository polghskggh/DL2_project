import torch
import torch.nn as nn



class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):
        return self.f(x)

    def forward(self, x, y=None):
        logits = self.classify(x)
        return (
            (-logits.logsumexp(1), logits)
            if y is None
            else (torch.gather(logits, 1, y.unsqueeze(1)), logits)
        )


class EnergyAdaptation(nn.Module):
    DEFAULT_CONFIG = {
        "steps": 1,
        "buffer_size": 10000,
        "sgld_steps": 20,
        "sgld_lr": 1.0,
        "sgld_std": 0.01,
        "reinit_freq": 0.05,
        "is_cond": False,
        "n_classes": 2,
    }

    def __init__(self, model: nn.Module, config: dict):
        super(EnergyAdaptation, self).__init__()
        self.energy_model = EnergyModel(model)
        self.config = config
        self.replay_buffer = []

        for key, default in self.DEFAULT_CONFIG.items():
            setattr(self, key, config.get(key, default))

        self.params, _ = self.__collect_params()
        self.optimizer = self.__setup_optimizer() if len(self.params) > 0 else None

    @torch.enable_grad()
    def forward(self, x, should_adapt=True):
        if should_adapt:
            self.energy_model.train()
            for _ in range(self.steps):
                _ = self.__forward_and_adapt(x)

        self.energy_model.eval()
        with torch.no_grad():
            return self.energy_model.classify(x)

    def __forward_and_adapt(self, x):
        batch_size, n_channels, n_timestamps = x.shape
        device = x.device

        y = None if not self.is_cond \
            else torch.randint(0, self.n_classes, (batch_size,), device=device)  # shouldn't this come from the actual dataset??

        x_fake, _ = self.__sample_q(
            batch_size=batch_size,
            series_length=n_timestamps,
            n_channels=n_channels,
            device=device,
            y=y,
        )

        energy_real, _ = self.energy_model(x)
        energy_fake, _ = self.energy_model(x_fake)

        loss = (energy_fake - energy_real).mean()
        loss.backward()

        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad()

        return self.energy_model.classify(x)

    @staticmethod
    def init_random(buffer_size, series_length=1000, n_channels=22):
        return torch.FloatTensor(buffer_size, n_channels, series_length).uniform_(-1, 1)

    def _sample_p_0(self, buffer_size, series_length, n_channels, device):
        random_samples = self.init_random(buffer_size, series_length, n_channels).to(device)

        if len(self.replay_buffer) == 0:
            return random_samples, None

        buffer_samples = torch.stack([
            self.replay_buffer[i] for i in torch.randint(0, len(self.replay_buffer), (buffer_size,))
        ]).to(device)

        choose_random = (torch.rand(buffer_size, 1, 1, device=device) < self.reinit_freq).float()
        samples = choose_random * random_samples + (1 - choose_random) * buffer_samples

        return samples, None

    def __sample_q(self, batch_size, series_length, n_channels, device, y=None):
        init_samples, _ = self._sample_p_0(
            buffer_size=batch_size,
            series_length=series_length,
            n_channels=n_channels,
            device=device,
        )

        init_samples = init_samples.clone().detach()
        x_k = torch.autograd.Variable(init_samples, requires_grad=True).to(device)

        for _ in range(self.sgld_steps):
            energy, _ = self.energy_model(x_k, y=y)
            grad = torch.autograd.grad(energy.sum(), [x_k], retain_graph=True)[0]
            with torch.no_grad():
                x_k += self.sgld_lr * grad + self.sgld_std * torch.randn_like(x_k)

        final_samples = x_k.detach()

        # Add new samples to buffer
        self.replay_buffer.extend([s.cpu() for s in final_samples])
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer = self.replay_buffer[-self.buffer_size:]

        return final_samples, init_samples

    def __setup_optimizer(self):
        optimizer = self.config.get("optimizer", None)
        kwargs = self.config.get("optimizer_kwargs", {})
        if optimizer == 'Adam':
            return torch.optim.Adam(
                self.params,
                lr=kwargs.get("lr", 1e-3),
                betas=(kwargs.get("beta", 0.9), 0.999),
                weight_decay=kwargs.get("weight_decay", 1e-4)
            )
        else:
            raise NotImplementedError

    def __collect_params(self):
        params = []
        names = []
        for nm, m in self.energy_model.f.named_modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias'] and p.requires_grad:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names
