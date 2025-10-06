"""Model helpers for custom classification heads."""

from __future__ import annotations

from typing import Dict, Optional

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertForSequenceClassification,
)


class DistilBertForSequenceClassificationGELU(DistilBertForSequenceClassification):
    """DistilBERT classifier that swaps ReLU for GELU in the pre-classifier head."""

    def forward(  # type: ignore[override]
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]
        pooled_output = hidden_state[:, 0]
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = torch.nn.functional.gelu(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return (loss,) + output if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


def load_sequence_classification_model(
    model_name: str,
    num_labels: Optional[int] = None,
    id2label: Optional[Dict[int, str]] = None,
    label2id: Optional[Dict[str, int]] = None,
    classifier_activation: str = "relu",
):
    """Factory loader that swaps in custom heads when requested in config."""
    config = AutoConfig.from_pretrained(model_name)
    if num_labels is not None:
        config.num_labels = num_labels
    if id2label is not None:
        config.id2label = id2label
    if label2id is not None:
        config.label2id = label2id

    if classifier_activation.lower() == "gelu" and config.model_type == "distilbert":
        model = DistilBertForSequenceClassificationGELU.from_pretrained(
            model_name,
            config=config,
        )
        model.config.architectures = ["DistilBertForSequenceClassificationGELU"]
        return model

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
    )
