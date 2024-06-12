import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig

class DebertaCRF(DebertaPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.deberta = DebertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)
        self.init_weights()

        # Initialize CRF transitions
        self.crf.transitions = nn.Parameter(torch.randn(config.num_labels, config.num_labels))
        self.crf.start_transitions = nn.Parameter(torch.randn(config.num_labels))
        self.crf.end_transitions = nn.Parameter(torch.randn(config.num_labels))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            log_likelihood, tags = self.crf(logits, labels), self.crf.decode(logits)
            loss = 0 - log_likelihood
        else:
            tags = self.crf.decode(logits)
        tags = torch.Tensor(tags)

        if not return_dict:
            output = (tags,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return loss, logits

# Example usage
model_checkpoint = 'microsoft/deberta-base'
num_labels = 6  # Set this to the number of labels in your dataset

config = DebertaConfig.from_pretrained(model_checkpoint)
config.num_labels = num_labels
model = DebertaCRF.from_pretrained(model_checkpoint, config=config, ignore_mismatched_sizes=True)
