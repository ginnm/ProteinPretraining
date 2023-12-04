from transformers import AutoTokenizer
import torch
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union
from transformers import DebertaModel, DebertaPreTrainedModel
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.activations import ACT2FN
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional, Tuple, Union


class MeanPooling(nn.Module):
    """Mean Pooling for sentence-level classification tasks."""

    def __init__(self):
        super().__init__()

    def forward(self, features, input_mask=None):
        if input_mask is not None:
            # Applying input_mask to zero out masked values
            masked_features = features * input_mask.unsqueeze(2)
            sum_features = torch.sum(masked_features, dim=1)
            mean_pooled_features = sum_features / input_mask.sum(dim=1, keepdim=True)
        else:
            mean_pooled_features = torch.mean(features, dim=1)
        return mean_pooled_features


class MeanPoolingProjection(nn.Module):
    """Mean Pooling with a projection layer for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        self.config = config

    def forward(self, mean_pooled_features):
        x = self.dropout(mean_pooled_features)
        x = self.dense(x)
        x = ACT2FN["gelu"](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MeanPoolingHead(nn.Module):
    """Mean Pooling Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.mean_pooling = MeanPooling()
        self.mean_pooling_projection = MeanPoolingProjection(config)

    def forward(self, features, input_mask=None):
        mean_pooling_features = self.mean_pooling(features, input_mask=input_mask)
        x = self.mean_pooling_projection(mean_pooling_features)
        return x


class AttentionPoolingHead(nn.Module):
    """Attention Pooling Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.scores = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Softmax(dim=1))
        self.dense = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, config.num_labels),
        )
        self.config = config

    def forward(self, features, input_mask=None):
        attention_scores = self.scores(features).transpose(1, 2)  # [B, 1, L]
        if input_mask is not None:
            # Applying input_mask to attention_scores
            attention_scores = attention_scores * input_mask.unsqueeze(1)
        context = torch.bmm(
            attention_scores, features
        ).squeeze()  # [B, 1, L] * [B, L, D] -> [B, 1, D]
        x = self.dense(context)
        return x


class MaskedConv1d(nn.Conv1d):
    """A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1dPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer = MaskedConv1d(config.hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_szie = x.shape[0]
        attn = self.layer(x)
        attn = attn.view(batch_szie, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(
                ~input_mask.view(batch_szie, -1).bool(), float("-inf")
            )
        attn = F.softmax(attn, dim=-1).view(batch_szie, -1, 1)
        out = (attn * x).sum(dim=1)
        return out


class Attention1dPoolingProjection(nn.Module):
    def __init__(self, config) -> None:
        super(Attention1dPoolingProjection, self).__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.final = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.relu(self.linear(x))
        x = self.final(x)
        return x


class Attention1dPoolingHead(nn.Module):
    """Outputs of the model with the attention1d"""

    def __init__(
        self, config
    ):  # [batch x sequence(751) x embedding (1280)] --> [batch x embedding] --> [batch x 1]
        super(Attention1dPoolingHead, self).__init__()
        self.attention1d = Attention1dPooling(config)
        self.attention1d_projection = Attention1dPoolingProjection(config)

    def forward(self, x, input_mask):
        x = self.attention1d(x, input_mask=input_mask.unsqueeze(-1))
        x = self.attention1d_projection(x)
        return x


class DebertClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = ACT2FN["gelu"](x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    
class DebertForSequenceClassification(DebertaPreTrainedModel):
    def __init__(self, config, pooling_head="mean", num_labels=1, is_ppi=False):
        super().__init__(config=config)
        self.config = config
        self.num_labels = num_labels
        self.pooling_head = pooling_head
        self.is_ppi = is_ppi
        self.deberta = DebertaModel(config)
        if pooling_head == "mean":
            if is_ppi:
                self.pooling = MeanPooling()
                self.projection = MeanPoolingProjection(config)
            else:
                self.classifier = MeanPoolingHead(config)
        elif pooling_head == "attention":
            self.classifier = AttentionPoolingHead(config)
        elif pooling_head == "attention1d":
            if is_ppi:
                self.pooling = Attention1dPooling(config)
                self.projection = Attention1dPoolingProjection(config)
            else:
                self.classifier = Attention1dPoolingHead(config)
        elif pooling_head == "cls":
            self.classifier = DebertClassificationHead(config)
        else:
            raise NotImplementedError(f"pooling head {pooling_head} not implemented")

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,

        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.Tensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        if self.pooling_head == "cls":
            logits = self.classifier(sequence_output)
        else:
            # mean pooling, attention pooling, attention1d pooling
            if self.is_ppi:
                sequence_output = self.pooling(sequence_output, attention_mask)
                pair_sequence_output = sequence_output.reshape(
                    -1, 2, sequence_output.shape[1]
                )
                pair_sum_sequence_output = torch.sum(pair_sequence_output, dim=1)
                logits = self.projection(pair_sum_sequence_output)
            else:
                logits = self.classifier(sequence_output, attention_mask)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (
                    labels.dtype == torch.long or labels.dtype == torch.int
                ):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                if self.num_labels == 1:
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits.squeeze(), labels.float())
                    logits = logits.squeeze()
                else:
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits, labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                # RuntimeError: result type Float can't be cast to the desired output type Long
                loss = loss_fct(logits, labels.float())
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    model = DebertForSequenceClassification.from_pretrained("AI4Protein/meer_base", pooling_head="attention1d", num_labels=1, is_ppi=False)
    tokenizer = AutoTokenizer.from_pretrained("AI4Protein/meer_base")
    sequences = [
        "MSKLHJSKLGJSGKLSJKHKLSHJ",
        "JKSLHGKSJGHSGJ"
    ]
    print(tokenizer.get_vocab())
    encoded_inputs = tokenizer(sequences, return_tensors="pt", padding=True)
    labels = torch.tensor([0, 1], dtype=torch.long)
    print(encoded_inputs)
    print(model(**encoded_inputs, labels=labels))