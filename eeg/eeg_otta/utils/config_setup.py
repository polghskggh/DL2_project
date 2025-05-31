import yaml
import os

from pathlib import Path

from .get_datamodule_cls import get_datamodule_cls
from .get_tta_cls import get_tta_cls
from eeg_otta.models import BaseNet

CHECKPOINT_PATH = os.path.join(Path(__file__).resolve().parents[2], "checkpoints")
CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[2], "configs")
LOG_DIR = os.path.join(Path(__file__).resolve().parents[2], "logs")

adaptation_method_file_config = {
    "tea": "tta_energy.yaml",
    "entropy_minimization": "tta_entropy_minimization.yaml",
    "source": "tta_no_adaptation.yaml",
}

def load_config(config):
    with open(os.path.join(CHECKPOINT_PATH, config["source_run"], "config.yaml")) as f:
        source_config = yaml.safe_load(f)

    for key, value in config.items():
        source_config[key] = value
    return source_config

def setup(config):
    # load source config
    datamodule_cls = get_datamodule_cls(config["dataset_name"])
    model_cls = BaseNet
    tta_cls = get_tta_cls(config["tta_method"])

    if config["subject_ids"] == "all":
        subject_ids = datamodule_cls.all_subject_ids
    else:
        subject_ids = [config["subject_ids"]]

    config["subject_ids"] = subject_ids

    if config["tta_config"]["alignment"]:
        config["preprocessing"]["alignment"] = False

    datamodule = datamodule_cls(config["preprocessing"], subject_ids=subject_ids)
    return model_cls, tta_cls, datamodule

def setup_config(dataset_name, dataset_setup, adaptation_method, seed, corruption_level=None):
    """
    Setup the configuration for the adaptation method.

    Args:
        dataset_name (str): Name of the dataset.
        dataset_setup (str): Setup configuration for the dataset.
        adaptation_method (str): Name of the adaptation method.
        seed (int): Random seed for reproducibility.
        corruption_level (int, optional): Level of corruption to apply. Defaults to None.

    Returns:
        dict: Configuration dictionary.
    """

    tta_config = adaptation_method_file_config.get(adaptation_method, False)
    if dataset_setup not in ["within", "loso"]:
        raise ValueError(f"Dataset setup {dataset_setup} is not supported. Supported setups are: ['within', 'loso']")
    if dataset_name not in ["2a", "2b"]:
        raise ValueError(f"Dataset name {dataset_name} is not supported. Supported datasets are: ['2a', '2b']")
    if tta_config is False:
        raise ValueError(f"Adaptation method {adaptation_method} is not supported. Supported methods are: {[_ for _ in adaptation_method_file_config.keys()]}")
    elif tta_config == "tta_energy.yaml":
        if dataset_name == '2a':
            tta_config = "tta_energy_2a.yaml"
        elif dataset_name == '2b':
            tta_config = "tta_energy_2b.yaml"

    with open(os.path.join(CONFIG_DIR, tta_config)) as f:
        config = yaml.safe_load(f)

    config["source_run"] = f"{dataset_name[1]}_{dataset_setup}"

    config = load_config(config)
    os.makedirs(os.path.join(LOG_DIR, config['source_run']), exist_ok=True)

    model_cls, tta_cls, datamodule = setup(config)
    datamodule.corruption_level = corruption_level

    save_dir = os.path.join(
        LOG_DIR,
        config['source_run'],
        f"{config['tta_config']['log_name']}{'_' + str(corruption_level) if corruption_level else ''}"
    )

    os.makedirs(save_dir, exist_ok=True)
    config['tta_config']['save_dir'] = save_dir
    config['seed'] = seed
    config['tta_config']['seed'] = seed

    return model_cls, tta_cls, datamodule, config