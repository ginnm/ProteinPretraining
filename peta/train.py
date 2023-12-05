from pytorch_lightning import Trainer
from argparse import ArgumentParser
from peta.dataset import *
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch
from peta.model import ProxyModel
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import seed_everything
import warnings

warnings.filterwarnings("ignore")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="fluorescence", choices=DATASETS)
    parser.add_argument("--split_method", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--model", type=str, default="AI4Protein/deep_base")
    parser.add_argument("--model_type", type=str, default="roformer")
    parser.add_argument("--pooling_head", type=str, default="mean")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="debug")
    parser.add_argument("--wandb_entity", type=str, default="matwings")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--finetune", type=str, default="all", choices=["all", "head", "lora"])

    # Trainer
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="gpu")
    parser.add_argument("--strategy", type=str, default="auto")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=5.0)
    parser.add_argument("--gradient_clip_algorithm", type=str, default="value")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", type=str, default="32")

    args = parser.parse_args()

    if args.tokenizer is None:
        args.tokenizer = args.model

    seed_everything(args.seed)
    return args


def get_dataset(args):
    if args.dataset == "fluorescence":
        dataset = FluorescenceDataset().to_datasets()
    elif args.dataset == "stability":
        dataset = StabilityDataset().to_datasets()
    elif args.dataset == "remote_homology":
        dataset = RemoteHomologyDetectionDataset().to_datasets(args.split_method)
    elif args.dataset == "meltome":
        dataset = MeltomeDataset().to_datasets(args.split_method)
    elif args.dataset == "gb1":
        dataset = Gb1Dataset().to_datasets(args.split_method)
    elif args.dataset == "aav":
        dataset = AAVDataset().to_datasets(args.split_method)
    elif args.dataset == "deepsol":
        dataset = DeepsolDataset().to_datasets()
    elif args.dataset == "esol":
        dataset = EsolDataset().to_datasets()
    elif args.dataset == "solmut_blat":
        dataset = SolmutBlatDataset().to_datasets()
    elif args.dataset == "solmut_cs":
        dataset = SolmutCsDataset().to_datasets()
    elif args.dataset == "solmut_lgk":
        dataset = SolmutLgkDataset().to_datasets()
    elif args.dataset == "deeploc-1":
        dataset = Deeploc1Dataset().to_datasets()
    elif args.dataset == "deeploc_binary":
        dataset = DeeplocBinaryDataset().to_datasets()
    elif args.dataset == "deeploc-2":
        dataset = Deeploc2Dataset().to_datasets(args.split_method)
    elif args.dataset == "deeploc_signal":
        dataset = DeeplocSignalDataset().to_datasets()
    elif args.dataset == "ppi_yeast":
        dataset = PPIYeastDataset().to_datasets()
    elif args.dataset == "ppi_shs27k":
        dataset = PPIShs27kDataset().to_datasets()
    elif args.dataset == "ppi_sun":
        dataset = PPISunDataset().to_datasets()
    else:
        raise ValueError("No such dataset")
    return dataset


def get_dataloader(tokenizer, args):
    
    dataset = get_dataset(args)
    from pytorch_lightning.utilities import rank_zero_info
    # STATS
    rank_zero_info(f"Dataset: {args.dataset}")
    rank_zero_info(f"Train: {len(dataset['train'])}")
    rank_zero_info(f"Valid: {len(dataset['valid'])}")
    rank_zero_info(f"Test: {len(dataset['test'])}")
    

    def collate_fn(batch):
        if "ppi" in args.dataset:
            sequences = []
            for example in batch:
                sequences.extend(example["sequence"])
        else:
            sequences = [example["sequence"] for example in batch]
        sequences = tokenizer(
            sequences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_length,
        )
        
        labels = [example["label"] for example in batch]
        sequences["labels"] = torch.tensor(labels)
        return sequences

    train_loader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    valid_loader = DataLoader(
        dataset["valid"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        dataset["test"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    return train_loader, valid_loader, test_loader


def main():
    args = parse_args()

    torch.set_float32_matmul_precision("high")
    # torch.set_bfloat16_matmul_precision("highest")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    train_loader, valid_loader, test_loader = get_dataloader(tokenizer, args)

    model = ProxyModel(
        model_path=args.model,
        model_type=args.model_type,
        pooling_head=args.pooling_head,
        is_ppi=True if "ppi" in args.dataset else False,
        num_labels=DATASET_TO_NUM_LABELS[args.dataset],
        problem_type=DATASET_TO_TASK[args.dataset],
        tokenizer=tokenizer,
        optim_args=args,
        metrics=DATASET_TO_METRICS[args.dataset],
    )

    monitor = DATSET_TO_MONITOR[args.dataset]
    m = args.model.split("/")[-1]
    ckpt_name = f"{args.dataset}-{args.split_method}-{m}-{args.pooling_head}" if args.split_method else f"{args.dataset}-{m}-{args.pooling_head}"
    if args.wandb_run_name is None:
        args.wandb_run_name = ckpt_name
    
    mode = "max"
    min_mode_dataset = ["esol"]
    if args.dataset in min_mode_dataset:
        mode = "min"
    
    model_checkpoint = ModelCheckpoint(
        "checkpoints/ft",
        monitor=DATSET_TO_MONITOR[args.dataset],
        mode=mode,
        filename=ckpt_name,
        verbose=True,
    )

    trainer = Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        num_nodes=args.num_nodes,
        logger=[WandbLogger(
            project=args.wandb_project, entity=args.wandb_entity, 
            name=args.wandb_run_name, config=vars(args)
            )] if args.wandb else None,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        deterministic=True,
        precision=args.precision,
        callbacks=[
            model_checkpoint,
            EarlyStopping(
                monitor=monitor, mode="max", patience=args.patience, verbose=True
            ),
        ],
        max_epochs=args.max_epochs,
    )
    trainer.fit(model, train_loader, valid_loader)
    model = ProxyModel.load_from_checkpoint(model_checkpoint.best_model_path)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
