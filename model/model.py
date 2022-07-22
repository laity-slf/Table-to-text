import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel


class RecordEncoding(BertPreTrainedModel):
    def __init__(self, config):
        super(RecordEncoding, self).__init__(config)

        self.config = config
        self.bert = BertModel(config)

        self.init_weights()

    def forward(self, input_ids, attention_mask, token_type_ids, output_attentions=False, output_hidden_states=False):
        return self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids,
                         output_attentions=output_attentions,
                         output_hidden_states=output_hidden_states)


class Regression(nn.Module):
    def __init__(self, d_model=768, num_layers=6, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, dim_feedforward=330, nhead=8
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pred_layer = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
            # nn.Linear(2*d_model,d_model)
        )

    def forward(self, x):
        # out: (length, batch size, d_model)
        out = x.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        # stats = out.mean(dim=1)
        # out: (batch, n_spks)
        out = self.pred_layer(out)
        return out
