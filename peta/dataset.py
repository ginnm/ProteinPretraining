### test
import os
import numpy as np
from torchmetrics.classification import Accuracy, AUROC
from torchmetrics.regression import MeanSquaredError
from torchmetrics.regression import SpearmanCorrCoef
from dataclasses import dataclass
from pathlib import Path
from datasets import load_dataset, Dataset

BASE_DIR = Path(__file__).resolve().parent.parent


def add_M(example):
    example["sequence"] = "M" + example["sequence"]
    return example


def min_max_norm_func(dataset):
    # min-max normalization
    max_label = max(
        [each[0] for each in dataset["train"]["label"]]
        + [each[0] for each in dataset["valid"]["label"]]
        + [each[0] for each in dataset["test"]["label"]]
    )
    min_label = min(
        [each[0] for each in dataset["train"]["label"]]
        + [each[0] for each in dataset["valid"]["label"]]
        + [each[0] for each in dataset["test"]["label"]]
    )

    def norm_func(example):
        example["label"] = [(example["label"][0] - min_label) / (max_label - min_label)]
        return example

    dataset = dataset.map(norm_func)
    return dataset


def z_score_norm_func(dataset):
    # z-score normalization
    data = np.array(
        [each[0] for each in dataset["train"]["label"]]
        + [each[0] for each in dataset["valid"]["label"]]
        + [each[0] for each in dataset["test"]["label"]]
    )
    mean = np.mean(data, )
    std = np.std(data,)

    def norm_func(example):
        example["label"] = [(example["label"][0] - mean) / std]
        return example

    dataset = dataset.map(norm_func)
    return dataset

DATASETS = [
    "fluorescence", "stability", "remote_homology", "gb1", "aav", "meltome", 
    "deepsol", "esol", "solmut_blat", "solmut_cs", "solmut_lgk", 
    "deeploc-1", "deeploc_binary", "deeploc-2", "deeploc_signal",
    "ppi_yeast", "ppi_shs27k", "ppi_sun"
]

DATASET_TO_TASK = {
    "fluorescence": "regression",
    "stability": "regression",
    "remote_homology": "single_label_classification",
    "meltome": "regression",
    "gb1": "regression",
    "aav": "regression",
    "deepsol": "single_label_classification",
    "esol": "regression",
    "solmut_blat": "regression",
    "solmut_cs": "regression",
    "solmut_lgk": "regression",
    "deeploc-1": "multi_label_classification",
    "deeploc_binary": "single_label_classification",
    "deeploc-2": "multi_label_classification",
    "deeploc_signal": "multi_label_classification",
    "ppi_yeast": "single_label_classification",
    "ppi_shs27k": "single_label_classification",
    "ppi_sun": "single_label_classification",
}

DATASET_TO_NUM_LABELS = {
    "fluorescence": 1,
    "stability": 1,
    "remote_homology": 1195,
    "meltome": 1,
    "gb1": 1,
    "aav": 1,
    "deepsol": 1,
    "esol": 1,
    "solmut_blat": 1,
    "solmut_cs": 1,
    "solmut_lgk": 1,
    "deeploc-1": 10,
    "deeploc_binary": 2, # "deeploc_binary" is the same as "deepsol"
    "deeploc-2": 10,
    "deeploc_signal": 9,
    "ppi_yeast": 2,
    "ppi_shs27k": 7,
    "ppi_sun": 2,
}

DATASET_TO_METRICS = {
    "fluorescence": [
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},  # valid
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},  # test
    ],
    "stability": [
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
    ],
    "remote_homology": [
        {"accuracy": Accuracy(task="multiclass", num_classes=1195)},
        {"accuracy": Accuracy(task="multiclass", num_classes=1195)},
    ],
    "meltome": [
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
    ],
    "gb1": [
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
    ],
    "aav": [
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
    ],
    "deepsol": [
        {"accuracy": Accuracy(task="binary"), "auroc" : AUROC(task="binary")},
        {"accuracy": Accuracy(task="binary"), "auroc" : AUROC(task="binary")},
    ],
    "esol": [
        {"mse": MeanSquaredError()},
        {"mse": MeanSquaredError()},
    ],
    "solmut_blat": [
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
    ],
    "solmut_cs": [
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
    ],
    "solmut_lgk": [
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
        {"mse": MeanSquaredError(), "spearman": SpearmanCorrCoef()},
    ],
    "deeploc-1": [
        {"accuracy": Accuracy(task="multilabel", num_labels=10)},
        {"accuracy": Accuracy(task="multilabel", num_labels=10)},
    ],
    "deeploc_binary": [
        {"accuracy": Accuracy(task="multiclass", num_classes=2)},
        {"accuracy": Accuracy(task="multiclass", num_classes=2)},
    ],
    "deeploc-2": [
        {"accuracy": Accuracy(task="multilabel", num_labels=10)},
        {"accuracy": Accuracy(task="multilabel", num_labels=10)},
    ],
    "deeploc_signal": [
        {"accuracy": Accuracy(task="multilabel", num_labels=9)},
        {"accuracy": Accuracy(task="multilabel", num_labels=9)},
    ],
    "ppi_yeast": [
        {"accuracy": Accuracy(task="multiclass", num_classes=2)},
        {"accuracy": Accuracy(task="multiclass", num_classes=2)},
    ],
    "ppi_shs27k": [
        {"accuracy": Accuracy(task="multiclass", num_classes=7)},
        {"accuracy": Accuracy(task="multiclass", num_classes=7)},
    ],
    "ppi_sun": [
        {"accuracy": Accuracy(task="multiclass", num_classes=2)},
        {"accuracy": Accuracy(task="multiclass", num_classes=2)},
    ],
}

DATSET_TO_MONITOR = {
    "fluorescence": "valid/spearman",
    "stability": "valid/spearman",
    "remote_homology": "valid/accuracy",
    "meltome": "valid/spearman",
    "gb1": "valid/spearman",
    "aav": "valid/spearman",
    "deepsol": "valid/auroc",
    "esol": "valid/mse",
    "solmut_blat": "valid/spearman",
    "solmut_cs": "valid/spearman",
    "solmut_lgk": "valid/spearman",
    "deeploc-1": "valid/accuracy",
    "deeploc_binary": "valid/accuracy",
    "deeploc-2": "valid/accuracy",
    "deeploc_signal": "valid/accuracy",
    "ppi_yeast": "valid/accuracy",
    "ppi_shs27k": "valid/accuracy",
    "ppi_sun": "valid/accuracy",
}


@dataclass
class FluorescenceDataset:
    task = "regression"
    train: str = "./ft_datasets/tape/fluorescence/fluorescence_train.json"
    valid: str = "./ft_datasets/tape/fluorescence/fluorescence_valid.json"
    test: str = "./ft_datasets/tape/fluorescence/fluorescence_test.json"

    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={"train": self.train, "test": self.test, "valid": self.valid},
        )
        dataset = dataset.remove_columns(["id", "num_mutations", "protein_length"])
        dataset = dataset.rename_column("primary", "sequence")
        dataset = dataset.rename_column("log_fluorescence", "label")
        # Add M for each sequence at the beginning
        dataset = dataset.map(add_M)
        dataset = z_score_norm_func(dataset)
        return dataset


@dataclass
class StabilityDataset:
    task = "regression"
    train: str = "./ft_datasets/tape/stability/stability_train.json"
    valid: str = "./ft_datasets/tape/stability/stability_valid.json"
    test: str = "./ft_datasets/tape/stability/stability_test.json"

    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={"train": self.train, "test": self.test, "valid": self.valid},
        )
        dataset = dataset.remove_columns(["id", "topology", "protein_length", "parent"])
        dataset = dataset.rename_column("primary", "sequence")
        dataset = dataset.rename_column("stability_score", "label")
        dataset = dataset.map(add_M)
        dataset = z_score_norm_func(dataset)
        return dataset


@dataclass
class RemoteHomologyDetectionDataset:
    task = "classification"
    base_dir = "./ft_datasets/deepsf"
    train: str = "./ft_datasets/deepsf/train.json"
    valid: str = "./ft_datasets/deepsf/valid.json"
    split_methods = ["fold_holdout", "family_holdout", "superfamily_holdout"]

    def to_datasets(self, split) -> Dataset:
        assert split in self.split_methods, f"split must be one of {self.split_methods}"
        dataset = load_dataset(
            "json",
            data_files={
                "train": self.train, 
                "test": os.path.join(self.base_dir, f"test_{split}.json"),
                "valid": self.valid
                },
        )
        # Fold label is the label, 1195 classes (0 - 1194)
        dataset = dataset.remove_columns(
            [
                "secondary_structure",
                "solvent_accessibility",
                "evolutionary",
                "superfamily_label",
                "class_label",
                "family_label",
                "protein_length",
                "id",
            ]
        )
        dataset = dataset.rename_column("fold_label", "label")
        dataset = dataset.rename_column("primary", "sequence")
        dataset = dataset.map(add_M)
        return dataset


@dataclass
class MeltomeDataset:
    task = "regression"
    base_dir: str = "./ft_datasets/flip/meltome"
    split_methods = ["human", "human_cell", "mixed_split"]

    def to_datasets(self, split_method) -> Dataset:
        assert split_method in self.split_methods, f"split_method must be one of {self.split_methods}"
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, split_method, "total.json"),
                "test": os.path.join(self.base_dir, split_method, "test.json"),
                "valid": os.path.join(self.base_dir, split_method, "valid.json"),
                },
        )

        # min-max normalization
        dataset = z_score_norm_func(dataset)
        return dataset


@dataclass
class Gb1Dataset:
    task = "regression"
    base_dir: str = "./ft_datasets/flip/gb1"
    split_methods = ["sampled", "one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high"]

    def to_datasets(self, split_method) -> Dataset:
        assert split_method in self.split_methods, f"split_method must be one of {self.split_methods}"
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, split_method, "train.json"),
                "test": os.path.join(self.base_dir, split_method, "test.json"),
                "valid": os.path.join(self.base_dir, split_method, "valid.json"),
                },
        )
        # dataset = z_score_norm_func(dataset)
        return dataset


@dataclass
class AAVDataset:
    task = "regression"
    base_dir: str = "ft_datasets/flip/aav"
    split_methods = ["des_mut", "mut_des", "sampled", "low_vs_high", "one_vs_many", "two_vs_many", "seven_vs_many"]

    def to_datasets(self, split_method) -> Dataset:
        assert split_method in self.split_methods, f"split_method must be one of {self.split_methods}"
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, split_method, "train.json"),
                "test": os.path.join(self.base_dir, split_method, "test.json"),
                "valid": os.path.join(self.base_dir, split_method, "valid.json"),
                },
        )
        dataset = z_score_norm_func(dataset)
        return dataset

@dataclass
class DeepsolDataset:
    task = "classification"
    base_dir: str = "ft_datasets/sol/deepsol"
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "total.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        return dataset

@dataclass
class EsolDataset:
    task = "regression"
    base_dir: str = "ft_datasets/sol/esol"
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        return dataset

@dataclass
class SolmutBlatDataset:
    task = "regression"
    base_dir: str = "ft_datasets/sol/soluprotmutdb/Beta-lactamase_TEM"
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        dataset = dataset.remove_columns(
            [
                "uniprot_id", "pdb_id", "host_cell", "host_strain",
                "system_type", "assay_type", "temperature", "ec_number",
                "spmdb_ac", "mutations"
            ]
        )
        dataset = z_score_norm_func(dataset)
        return dataset

@dataclass
class SolmutCsDataset:
    task = "regression"
    base_dir: str = "ft_datasets/sol/soluprotmutdb/chalcone_synthase"
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        dataset = dataset.remove_columns(
            [
                "uniprot_id", "pdb_id", "host_cell", "host_strain",
                "system_type", "assay_type", "temperature", "ec_number",
                "spmdb_ac", "mutations"
            ]
        )
        dataset = z_score_norm_func(dataset)
        return dataset

@dataclass
class SolmutLgkDataset:
    task = "regression"
    base_dir: str = "ft_datasets/sol/soluprotmutdb/Levoglucosan_kinase"
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        dataset = dataset.remove_columns(
            [
                "uniprot_id", "pdb_id", "host_cell", "host_strain",
                "system_type", "assay_type", "temperature", "ec_number",
                "spmdb_ac", "mutations"
            ]
        )
        dataset = z_score_norm_func(dataset)
        return dataset


@dataclass
class Deeploc1Dataset:
    task = "classification"
    base_dir: str = "ft_datasets/deeploc-1/location"
    # 10 classes
    label_names = ["Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion", "Cell.membrane", "Endoplasmic.reticulum", "Plastid", "Golgi.apparatus", "Lysosome/Vacuole", "Peroxisome"]
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        dataset = dataset.remove_columns(
            [
                "uniprot_id",
                "location",
            ]
        )
        return dataset

@dataclass
class DeeplocBinaryDataset:
    task = "classification"
    base_dir: str = "ft_datasets/deeploc-1/binary"
    # 2 classes
    label_names = ["M", "S"]
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        dataset = dataset.remove_columns(
            [
                "uniprot_id",
                "location",
            ]
        )
        return dataset


@dataclass
class Deeploc2Dataset:
    task = "classification"
    base_dir: str = "ft_datasets/deeploc-2/location"
    # 10 classes
    label_names = ["Cytoplasm","Nucleus","Extracellular","Cell membrane","Mitochondrion","Plastid","Endoplasmic reticulum","Lysosome/Vacuole","Golgi apparatus","Peroxisome"]
    split_methods = ["test", "hpa_test"]
    
    def to_datasets(self, split_method) -> Dataset:
        assert split_method in self.split_methods, f"split_method must be one of {self.split_methods}"
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, f"{split_method}.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        return dataset

@dataclass
class DeeplocSignalDataset:
    task = "classification"
    base_dir: str = "ft_datasets/deeploc-2/signal"
    # 9 classes
    label_names = ["MT", "SP", "GPI", "NLS", "PTS", "CH", "NES", "TH", "TM"]
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        dataset.remove_columns(
            [
                "uniprot_id",
                "kingdom",
            ]
        )
        return dataset

@dataclass
class PPIYeastDataset:
    task = "classification"
    base_dir: str = "ft_datasets/PPI/yeast"
    # 2 classes
    label_names = ["0", "1"]
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        dataset.remove_columns(
            [
                "uniprot_id",
            ]
        )
        return dataset

@dataclass
class PPIShs27kDataset:
    task = "classification"
    base_dir: str = "ft_datasets/PPI/shs27k"
    # 7 classes
    label_names = ["reaction", "activation", "catalysis", "binding", "ptmod", "inhibition", "expression"]
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        return dataset

@dataclass
class PPISunDataset:
    task = "classification"
    base_dir: str = "ft_datasets/PPI/sun"
    # 2 classes
    label_names = ["0", "1"]
    
    def to_datasets(self) -> Dataset:
        dataset = load_dataset(
            "json",
            data_files={
                "train": os.path.join(self.base_dir, "train.json"),
                "test": os.path.join(self.base_dir, "test.json"),
                "valid": os.path.join(self.base_dir, "val.json"),
                },
        )
        dataset.remove_columns(
            [
                "id",
            ]
        )
        return dataset