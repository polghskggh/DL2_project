from eeg_otta.tta import Norm, OnlineAlignment, EntropyMinimization, EnergyAdaptation, NoAdaptation


def get_tta_cls(tta_method: str):
    if tta_method == "alignment":
        return OnlineAlignment
    elif tta_method == "norm":
        return Norm
    elif tta_method == "entropy_minimization":
        return EntropyMinimization
    elif tta_method == "energy_adaptation":
        return EnergyAdaptation
    elif tta_method == "no_adaptation":
        return NoAdaptation
    else:
        raise NotImplementedError
