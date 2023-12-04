from peta.custom_ankh_model import AnkhForSequenceClassification
from peta.custom_debert import DebertForSequenceClassification
from peta.custom_esm_model import EsmForSequenceClassification
from peta.custom_roformer_model import RoFormerForSequenceClassification
import pytorch_lightning as pl
import torch
import torch.utils.checkpoint
from peft import get_peft_model, LoraConfig



class ProxyModel(pl.LightningModule):
    def __init__(
        self,
        model_type="esm",
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
        if model_type == "esm":
            self.model = EsmForSequenceClassification.from_pretrained(
                model_path, pooling_head=pooling_head, num_labels=num_labels, is_ppi=is_ppi
            )
        elif model_type == "debert":
            self.model = DebertForSequenceClassification.from_pretrained(
                model_path, pooling_head=pooling_head, num_labels=num_labels, is_ppi=is_ppi
            )
        elif model_type == "roformer":
            self.model = RoFormerForSequenceClassification.from_pretrained(
                model_path, pooling_head=pooling_head, num_labels=num_labels, is_ppi=is_ppi
            )
        elif model_type == "ankh":
            self.model = AnkhForSequenceClassification.from_pretrained(
                model_path, pooling_head=pooling_head, num_labels=num_labels, is_ppi=is_ppi
            )
        self.model.config.problem_type = problem_type
        self.model.config.num_labels = num_labels
        self.optmi_args = optim_args
        self.valid_metrics, self.test_metrics = metrics
        self.valid_metrics = torch.nn.ModuleDict(self.valid_metrics)
        self.test_metrics = torch.nn.ModuleDict(self.test_metrics)
        self.lr = optim_args.lr
        self.num_labels = num_labels
        self.problem_type = problem_type
        self.model_type = model_type

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
            logits = outputs.logits
            if self.num_labels == 2 and self.problem_type == "classification":
                logits = torch.sigmoid(outputs.logits)
            elif self.num_labels > 2 and self.problem_type == "classification":
                logits = torch.softmax(outputs.logits, dim=-1)
            self.log(
                f"valid/{name}",
                metric(logits, batch["labels"]),
                logger=True,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

    def test_step(self, batch, *args, **kwargs):
        outputs = self.model(**batch)
        if self.num_labels == 2 and self.problem_type == "classification":
            logits = torch.sigmoid(outputs.logits)
        elif self.num_labels > 2 and self.problem_type == "classification":
            logits = torch.softmax(outputs.logits, dim=-1)
        for name, metric in self.test_metrics.items():
            self.log(
                f"test/{name}",
                metric(logits, batch["labels"]),
                logger=True,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

    def configure_optimizers(self):
        if self.optmi_args.finetune == "head":
            if self.model_type == "esm":
                module = self.model.esm
            elif self.model_type == "debert":
                module = self.model.deberta
            elif self.model_type == "roformer":
                module = self.model.roformer
            elif self.model_type == "ankh":
                module = self.model.transformer
            for param in module.parameters():
                param.requires_grad = False
        elif self.optmi_args.finetune == "all":
            pass
        elif self.optmi_args.finetune == "lora":
            raise NotImplementedError
            peft_config = LoraConfig(
                target_modules=["k", "q", ],
                inference_mode=False,
                r=4,
                lora_alpha=8,
                lora_dropout=0.1,
            )
            module = get_peft_model(self.model.transformer, peft_config)
            self.model.transformer.print_trainable_parameters()
            
        else:
            raise ValueError(f"finetune={self.optmi_args.finetune} not supported")

        optimizer = torch.optim.AdamW(
            [p for p in self.parameters() if p.requires_grad],
            lr=self.lr,
            weight_decay=self.optmi_args.weight_decay,
        )
        return optimizer


