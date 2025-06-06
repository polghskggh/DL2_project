

from typing import Optional

import numpy as np
from torch.utils.data.dataloader import DataLoader

from .base import BaseDataModule
from ..utils.alignment import align
from ..utils.load_bcic import load_bcic


class BCICIV2aWithinSubject(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    corruption_level = None

    def __init__(self, preprocessing_dict: dict, subject_ids: list[int]):
        super(BCICIV2aWithinSubject, self).__init__(preprocessing_dict, subject_ids)

    def prepare_data(self) -> None:
        # for WithinSubject: only load one subject at a time
        self.dataset, self.info = load_bcic(
            subject_ids=[self.subject_id], dataset="2a",
            preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        # split the data
        splitted_ds = self.dataset.split("session")
        train_dataset, test_dataset = splitted_ds["session_T"], splitted_ds["session_E"]

        # load the data
        X = np.concatenate(
            [run.windows.load_data()._data for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for run in train_dataset.datasets], axis=0)
        X_test = np.concatenate(
            [run.windows.load_data()._data for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)

        def corrupt(x, severity=5):
            stds = [0.1, 1, 10, 20, 40]
            std = stds[severity - 1]

            noise = np.random.randn(*x.shape) * std
            corrupted = x + noise

            return corrupted

        # corrupt data
        if self.corruption_level:
            print('corrupting data level:', self.corruption_level)
            X_test = corrupt(X_test, severity=self.corruption_level)

        # align data
        X, X_test = align(self.preprocessing_dict["alignment"], X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2bWithinSubject(BaseDataModule):
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    corruption_level = None

    def __init__(self, preprocessing_dict: dict, subject_ids: list[int]):
        super(BCICIV2bWithinSubject, self).__init__(preprocessing_dict, subject_ids)

    def prepare_data(self) -> None:
        self.dataset, self.info = load_bcic(subject_ids=[self.subject_id], dataset="2b",
                                 preprocessing_dict=self.preprocessing_dict)

    def setup(self, stage: Optional[str] = None) -> None:
        # split the data
        splitted_ds = self.dataset.split("session")
        train_datasets = [splitted_ds[f"session_{session}"] for session in [0, 1, 2]]
        test_datasets = [splitted_ds[f"session_{session}"] for session in [3, 4]]

        # load the data
        X = np.concatenate(
            [run.windows.load_data()._data for train_dataset in train_datasets for run
             in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_test = np.concatenate(
            [run.windows.load_data()._data for test_dataset in test_datasets for run in
             test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                 test_dataset.datasets], axis=0)

        def corrupt(x, severity=5):
            stds = [0.1, 1, 10, 20, 40]
            std = stds[severity - 1]

            noise = np.random.randn(*x.shape) * std
            corrupted = x + noise

            return corrupted

        # corrupt data
        if self.corruption_level:
            print('corrupting data level:', self.corruption_level)
            X_test = corrupt(X_test, severity=self.corruption_level)

        # scale and align data
        X, X_test = align(self.preprocessing_dict["alignment"], X, X_test)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)


class BCICIV2aLOSO(BaseDataModule):
    val_dataset = None
    calibration_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["feet", "hand(L)", "hand(R)", "tongue"]
    prepare_called = False
    train_individual = False
    corruption_level = None

    def __init__(self, preprocessing_dict: dict, subject_ids: list[int]):
        super(BCICIV2aLOSO, self).__init__(preprocessing_dict, subject_ids)

    def prepare_data(self) -> None:
        # for LOSO: load all subject_ids once
        if not self.prepare_called:
            self.dataset, self.info = load_bcic(
                subject_ids=self.all_subject_ids, dataset="2a",
                preprocessing_dict=self.preprocessing_dict)
            self.prepare_called = True

    def setup(self, stage: Optional[str] = None) -> None:
        # split the data
        if stage == 'test':
            self.update_test_set()
        else:
            splitted_ds = self.dataset.split("subject")
            if self.train_individual:
                print('training for one subject')
                test_subjects = [
                    subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
                test_datasets_T = [splitted_ds[str(subj_id)].split("session")["session_T"]
                                  for subj_id in test_subjects]
                test_datasets = [splitted_ds[str(subj_id)].split("session")["session_E"]
                                for subj_id in test_subjects]
                train_subjects = [self.subject_id]
                train_datasets = splitted_ds[str(self.subject_id)].split("session")["session_T"]
                val_datasets = splitted_ds[str(self.subject_id)].split("session")["session_E"]

                # load the data
                X_test_T = np.concatenate([run.windows._data for test_dataset in
                                    test_datasets_T for run in test_dataset.datasets], axis=0)
                y_test_T = np.concatenate([run.y for test_dataset in test_datasets_T for run in
                                    test_dataset.datasets], axis=0)

                X_test = np.concatenate([run.windows._data for test_dataset in
                                           test_datasets for run in test_dataset.datasets], axis=0)
                y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                           test_dataset.datasets], axis=0)

                X = np.concatenate([run.windows._data for run in
                                           train_datasets.datasets], axis=0)
                y = np.concatenate([run.y for run in train_datasets.datasets], axis=0)

                X_val = np.concatenate([run.windows._data for run in val_datasets.datasets],
                                        axis=0)
                y_val = np.concatenate([run.y for run in val_datasets.datasets], axis=0)

                train_domains = np.concatenate(
                    [[subject_id] * ds.cummulative_sizes[-1] for (ds, subject_id) in
                     zip([train_datasets], train_subjects)])
                val_domains = np.concatenate(
                    [[subject_id] * ds.cummulative_sizes[-1] for (ds, subject_id) in
                     zip([val_datasets], train_subjects)])
            else:
                train_subjects = [
                    subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]
                train_datasets = [splitted_ds[str(subj_id)].split("session")["session_T"]
                                      for subj_id in train_subjects]
                val_datasets = [splitted_ds[str(subj_id)].split("session")["session_E"]
                                  for subj_id in train_subjects]
                test_dataset_T = splitted_ds[str(self.subject_id)].split("session")["session_T"]
                test_dataset = splitted_ds[str(self.subject_id)].split("session")["session_E"]

                # load the data
                X = np.concatenate([run.windows._data for train_dataset in
                                    train_datasets for run in train_dataset.datasets], axis=0)
                y = np.concatenate([run.y for train_dataset in train_datasets for run in
                                    train_dataset.datasets], axis=0)
                X_val = np.concatenate([run.windows._data for val_dataset in
                                    val_datasets for run in val_dataset.datasets], axis=0)
                y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                                    val_dataset.datasets], axis=0)
                X_test_T = np.concatenate([run.windows._data for run in
                                           test_dataset_T.datasets], axis=0)
                y_test_T = np.concatenate([run.y for run in test_dataset_T.datasets], axis=0)
                X_test = np.concatenate([run.windows._data for run in test_dataset.datasets],
                                        axis=0)
                y_test = np.concatenate([run.y for run in test_dataset.datasets], axis=0)
                train_domains = np.concatenate(
                    [[subject_id] * ds.cummulative_sizes[-1] for (ds, subject_id) in
                     zip(train_datasets, train_subjects)])
                val_domains = np.concatenate(
                    [[subject_id] * ds.cummulative_sizes[-1] for (ds, subject_id) in
                     zip(val_datasets, train_subjects)])

            # align data
            X, X_val = align(self.preprocessing_dict["alignment"], X, X_val,
                             train_domains=train_domains, test_domains=val_domains)
            _, X_test = align(self.preprocessing_dict["alignment"], X_test_T, X_test)

            def corrupt(x, severity=5):
                stds = [0.1, 1, 10, 20, 40]
                std = stds[severity - 1]

                noise = np.random.randn(*x.shape) * std
                corrupted = x + noise

                return corrupted

            # corrupt data
            if self.corruption_level:
                print('corrupting data level:', self.corruption_level)
                X_test = corrupt(X_test, severity=self.corruption_level)

            # make datasets
            self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
            self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
            self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
            self.calibration_dataset = BaseDataModule._make_tensor_dataset(X_test_T,
                                                                           y_test_T)

            print(f'num train {len(self.train_dataset)}')
            print(f'num val {len(self.val_dataset)}')
            print(f'num test {len(self.test_dataset)}')

    def update_test_set(self):
        splitted_ds = self.dataset.split("subject")
        test_set = splitted_ds[str(self.subject_id)].split("session")["session_E"]
        X_test = np.concatenate([run.windows._data for run in test_set.datasets],
                               axis=0)
        y_test = np.concatenate([run.y for run in test_set.datasets], axis=0)

        test_set_T = splitted_ds[str(self.subject_id)].split("session")["session_T"]
        X_test_T = np.concatenate([run.windows._data for run in test_set_T.datasets],
                                axis=0)

        _, X_test = align(self.preprocessing_dict["alignment"], X_test_T, X_test)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        print(f'num test {len(self.test_dataset)}')
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])

    def calibration_dataloader(self) -> DataLoader:
        return DataLoader(self.calibration_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])


class BCICIV2bLOSO(BaseDataModule):
    val_dataset = None
    calibration_dataset = None
    all_subject_ids = list(range(1, 10))
    class_names = ["hand(L)", "hand(R)"]
    prepare_called = False
    corruption_level = None

    def __init__(self, preprocessing_dict: dict, subject_ids: list[int]):
        super(BCICIV2bLOSO, self).__init__(preprocessing_dict, subject_ids)

    def prepare_data(self) -> None:
        # for LOSO: load all subject_ids once
        if not self.prepare_called:
            self.dataset, self.info = load_bcic(
                subject_ids=self.all_subject_ids, dataset="2b",
                preprocessing_dict=self.preprocessing_dict)
            self.prepare_called = True

    def setup(self, stage: Optional[str] = None) -> None:
        # split the data
        splitted_ds = self.dataset.split("subject")
        train_subjects = [
            subj_id for subj_id in self.all_subject_ids if subj_id != self.subject_id]

        train_datasets = [
            splitted_ds[str(subj_id)].split("session")[f"session_{session}"] for
            subj_id in train_subjects for session in [0, 1, 2]]
        val_datasets = [
            splitted_ds[str(subj_id)].split("session")[f"session_{session}"] for
            subj_id in train_subjects for session in [3, 4]]
        test_datasets_cal = [
            splitted_ds[str(self.subject_id)].split("session")[f"session_{session}"]
            for session in [2]]
        test_datasets = [
            splitted_ds[str(self.subject_id)].split("session")[f"session_{session}"]
            for session in [3, 4]]
        train_domains = np.concatenate(
            [[subject_id] * ds.cummulative_sizes[-1] for (ds, subject_id) in
             zip(train_datasets,
                 [subj_id for subj_id in train_subjects for _ in range(3)])])
        val_domains = np.concatenate(
            [[subject_id] * ds.cummulative_sizes[-1] for (ds, subject_id) in
             zip(val_datasets,
                 [subj_id for subj_id in train_subjects for _ in range(2)])])

        # load the data
        X = np.concatenate([run.windows._data for train_dataset in
                            train_datasets for run in train_dataset.datasets], axis=0)
        y = np.concatenate([run.y for train_dataset in train_datasets for run in
                            train_dataset.datasets], axis=0)
        X_val = np.concatenate([run.windows._data for val_dataset in
                            val_datasets for run in val_dataset.datasets], axis=0)
        y_val = np.concatenate([run.y for val_dataset in val_datasets for run in
                            val_dataset.datasets], axis=0)
        X_test_cal = np.concatenate([run.windows._data for test_dataset_cal in
                                     test_datasets_cal for run in
                                     test_dataset_cal.datasets], axis=0)
        y_test_cal = np.concatenate([run.y for test_dataset_cal in test_datasets_cal
                                     for run in test_dataset_cal.datasets], axis=0)
        X_test = np.concatenate([run.windows._data for test_dataset in test_datasets
                                 for run in test_dataset.datasets], axis=0)
        y_test = np.concatenate([run.y for test_dataset in test_datasets for run in
                                 test_dataset.datasets], axis=0)

        # align data
        X, X_val = align(self.preprocessing_dict["alignment"], X, X_val,
                         train_domains=train_domains, test_domains=val_domains)
        _, X_test = align(self.preprocessing_dict["alignment"], X_test_cal, X_test)

        def corrupt(x, severity=5):
            stds = [0.1, 1, 10, 20, 40]
            std = stds[severity - 1]

            noise = np.random.randn(*x.shape) * std
            corrupted = x + noise

            return corrupted

        # corrupt data
        if self.corruption_level:
            print('corrupting data level:', self.corruption_level)
            X_test = corrupt(X_test, severity=self.corruption_level)

        # make datasets
        self.train_dataset = BaseDataModule._make_tensor_dataset(X, y)
        self.val_dataset = BaseDataModule._make_tensor_dataset(X_val, y_val)
        self.test_dataset = BaseDataModule._make_tensor_dataset(X_test, y_test)
        self.calibration_dataset = BaseDataModule._make_tensor_dataset(X_test_cal,
                                                                       y_test_cal)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])

    def calibration_dataloader(self) -> DataLoader:
        return DataLoader(self.calibration_dataset,
                          batch_size=self.preprocessing_dict["batch_size"])
