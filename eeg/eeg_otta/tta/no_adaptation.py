import mne
import torch.nn as nn
import torch.jit

from .alignment import OnlineAlignment
from .base import TTAMethod


class NoAdaptation(TTAMethod):
    def __init__(self, model: nn.Module, config: dict, info: mne.Info):
        super(NoAdaptation, self).__init__(model, config, info)

    def forward_sliding_window(self, x):
        if self.config.get("alignment", False):
            # align data
            x = OnlineAlignment.align_data(
                x, self.config.get("alignment"),
                self.config.get("averaging_method", "equal"),
                self.config.get("align_alpha", None))
        outputs = self.model(x)
        return outputs

    @torch.enable_grad()  # ensure grads in possible no grad context for testing
    def forward_and_adapt(self, x):
        if self.config.get("alignment", False):
            # align data
            x = OnlineAlignment.align_data(
                x, self.config.get("alignment"),
                self.config.get("averaging_method", "equal"),
                self.config.get("align_alpha", None))

        outputs = self.model(x)
        return outputs

    def configure_model(self):
        self.model.eval()  # eval mode to avoid using dropout during test-time
        self.model.requires_grad_(True)
        for nm, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
