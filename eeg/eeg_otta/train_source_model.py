from argparse import ArgumentParser
from datetime import datetime
import os
from pathlib import Path
import yaml

import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from eeg_otta.models import BaseNet
from eeg_otta.utils.get_datamodule_cls import get_datamodule_cls
from eeg_otta.utils.seed import seed_everything

CHECKPOINT_PATH = os.path.join(Path(__file__).resolve().parents[1], "checkpoints")
CONFIG_DIR = os.path.join(Path(__file__).resolve().parents[1], "configs")
DEFAULT_CONFIG = "bcic2a_loso_basenet.yaml"


def train_source_model(config):
    # get datamodule_cls and model_cls
    model_cls = BaseNet
    datamodule_cls = get_datamodule_cls(dataset_name=config["dataset_name"])

    if config["subject_ids"] == "all":
        subject_ids = datamodule_cls.all_subject_ids
    elif isinstance(config["subject_ids"], int):
        subject_ids = [config["subject_ids"]]
    else:
        subject_ids = config["subject_ids"]
    datamodule = datamodule_cls(config["preprocessing"], subject_ids=subject_ids)

    test_accs = []
    now = datetime.now()
    run_name = f"src-{config['dataset_name']}" + now.strftime("_%Y-%m-%d_%H-%M-%S")

    # save config
    os.makedirs(os.path.join(CHECKPOINT_PATH, run_name), exist_ok=False)
    with open(os.path.join(CHECKPOINT_PATH, run_name, "config.yaml"), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    for subject_id in subject_ids:
        seed_everything(config["seed"])

        if (wandb_c := config.get("wandb")) is not None and wandb_c.get("project") is not None:
            logger = WandbLogger(
                project=wandb_c["project"],
                entity=wandb_c.get("entity"),
                name=run_name,
                config=config,
            )
        else:
            logger = None
        
        # set up the trainer
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join(CHECKPOINT_PATH, run_name, str(subject_id)),
            filename="model",
            monitor="val_acc",
            mode='max',
            save_top_k=1,
            verbose=True
        )
        callbacks = [checkpoint_cb]
        if config.get("early_stopping"):
            callbacks.append(EarlyStopping(monitor="val_acc", patience=25, mode="max"))
        
        trainer = Trainer(
            callbacks=callbacks,
            max_epochs=config["max_epochs"],
            logger=logger,
            num_sanity_val_steps=0
        )

        # set subject_id
        datamodule.subject_id = subject_id
        datamodule.train_individual = config["train_individual"]
        # train model
        model = model_cls(**config["model_kwargs"], max_epochs=config["max_epochs"])
        trainer.fit(model, datamodule=datamodule)

        # test model
        for i in datamodule_cls.all_subject_ids:
            if i not in subject_ids:
                datamodule.subject_id = i
                test_results = trainer.test(model, datamodule)
                test_accs.append(test_results[0]["test_acc"])

                print(f"source accuracy subject {i}: {100 *test_accs[-1]:.2f}%")
        
        if config["train_individual"]:
            break

    print(f"source accuracy: {100 *np.mean(test_accs):.2f}%")


if __name__ == "__main__":
    # parse arguments
    parser = ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()

    # load config
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = yaml.safe_load(f)

    train_source_model(config)
