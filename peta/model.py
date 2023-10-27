from peta.custom_roformer_model import RoFormerForSequenceClassification
import pytorch_lightning as pl
import torch


class ProxyModel(pl.LightningModule):
    def __init__(
        self,
        model_path="AI4Protein/deep_base",
        pooling_head="mean",
        num_labels=2,
        is_ppi=False,
        problem_type="classification",
        optim_args=None,
        metrics=(None, None),
        tokenizer=None,
    ):
        super().__init__()
        self.model = RoFormerForSequenceClassification.from_pretrained(
            model_path, pooling_head=pooling_head, num_labels=num_labels, is_ppi=is_ppi
        )
        self.model.config.problem_type = problem_type
        self.model.config.num_labels = num_labels
        self.optmi_args = optim_args
        self.valid_metrics, self.test_metrics = metrics
        self.valid_metrics = torch.nn.ModuleDict(self.valid_metrics)
        self.test_metrics = torch.nn.ModuleDict(self.test_metrics)
        self.lr = optim_args.lr

        self.save_hyperparameters(
            ignore=[
                "tokenizer",
            ]
        )

    def training_step(self, batch, *args, **kwargs):
        outputs = self.model(**batch)
        self.log(
            "train/loss",
            outputs.loss,
            logger=True,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "train/lr", lr, logger=True, on_step=True, on_epoch=False, prog_bar=True
        )
        self.log(
            "train/step",
            self.global_step,
            logger=True,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )
        return outputs.loss

    def validation_step(self, batch, *args, **kwargs):
        outputs = self.model(**batch)
        for name, metric in self.valid_metrics.items():
            self.log(
                f"valid/{name}",
                metric(outputs.logits, batch["labels"]),
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def test_step(self, batch, *args, **kwargs):
        outputs = self.model(**batch)
        for name, metric in self.test_metrics.items():
            self.log(
                f"test/{name}",
                metric(outputs.logits, batch["labels"]),
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        if self.optmi_args.finetune == "head":
            for param in self.model.roformer.parameters():
                param.requires_grad = False
        elif self.optmi_args.finetune == "all":
            pass
        else:
            raise ValueError(f"finetune={self.optmi_args.finetune} not supported")

        optimizer = torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.optmi_args.weight_decay,
        )
        return optimizer