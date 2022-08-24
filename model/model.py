import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel, BertConfig


# 定义model
class Bert_Model(nn.Module):
    def __init__(self, bert_path, classes=1):
        super(Bert_Model, self).__init__()
        self.config = BertConfig.from_pretrained(bert_path)  # 导入模型超参数
        self.bert = BertModel.from_pretrained(bert_path)  # 加载预训练模型权重
        self.fc = nn.Sequential(
            nn.Linear(self.config.hidden_size, 1),
            nn.ReLU(),
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, record_pos=None):
        batch = input_ids.size(0)
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        # 去 前22个record的对应输出向量
        out = outputs[0][record_pos, :]
        out_record = out.view(batch, -1, 768)[:, :20, :]  # 池化后的输出 [bs,22,config.hidden_size]
        logit = self.fc(out_record)  # [bs, classes]
        return logit.squeeze()


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


class linear_model(nn.Module):
    def __init__(self, input_dim, out_dim, dropout=0.2):
        super(linear_model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
            nn.Dropout(dropout),
            nn.Sigmoid(),)

    def forward(self, x):
        # batch * output_dim
        out = self.fc(x)

        return out
