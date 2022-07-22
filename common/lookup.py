import torch
import torch.nn as nn
from tqdm import tqdm


def get_record_ids(records, tokenizer):
    record_ids = []
    record_lens = []
    max_len = 0
    for record in records:
        record_token_ids = tokenizer(record)["input_ids"]  # special tokens added automatically
        record_len = len(record_token_ids)
        max_len = max(max_len, record_len)
        record_ids.append(record_token_ids)
        record_lens.append(record_len)

    record_ids_padded = []
    for record_item_ids in record_ids:
        item_len = len(record_item_ids)
        padding = [0] * (max_len - item_len)
        record_ids_padded.append(record_item_ids + padding)
    record_ids_padded = torch.tensor(record_ids_padded, dtype=torch.long)

    return record_ids_padded, record_lens


def get_record_lookup(records, scores, tokenizer, encoder, device, use_layernorm=True):
    model_output_dim = encoder.config.hidden_size
    record_lookup = nn.Embedding(len(records), model_output_dim)

    encoder.eval()
    LN = nn.LayerNorm(model_output_dim, elementwise_affine=False)

    # get record ids
    record_ids, record_lens = get_record_ids(records, tokenizer)

    # encoding
    record_type_ids = torch.zeros(record_ids.size(), dtype=torch.long)
    record_mask = record_ids > 0
    hid_record = encoder(record_ids, record_mask, record_type_ids)[0]
    hid_record = hid_record.detach()
    # remove [CLS] and [SEP]
    for lb in range(record_ids.size(0)):
        record_mask[lb, 0] = False
        record_mask[lb, record_lens[lb] - 1] = False
    expanded_record_mask = record_mask.view(-1, record_ids.size(1), 1).expand(hid_record.size()).float()
    hid_record = torch.mul(hid_record, expanded_record_mask)
    masked_record_len = torch.sum(expanded_record_mask, 1, True).repeat(1, record_ids.size(1), 1)
    hid_record = torch.mean(torch.true_divide(hid_record, masked_record_len), 1)
    if use_layernorm:
        hid_record = LN(hid_record)
    record_lookup = nn.Embedding.from_pretrained(hid_record, freeze=True).to(device)

    return record_lookup


def get_record_lookup_from_first_token(records, encoder, scores, tokenizer, device, use_layernorm=False):
    model_output_dim = encoder.config.hidden_size
    # record_lookup = nn.Embedding(len(records), model_output_dim)
    encoder.to(device)
    encoder.eval()
    hid_records = []
    LN = nn.LayerNorm(model_output_dim, elementwise_affine=False)

    # max_length_records = max([len(r) for r in records])
    max_length_records = 660
    for record in tqdm(records, desc='正在创建字典'):
        # get record ids
        record_ids, record_lens = get_record_ids(record, tokenizer)
        # encoding
        record_type_ids = torch.zeros(record_ids.size(), dtype=torch.long)
        record_mask = (record_ids > 0)
        hid_record = encoder(record_ids.to(device), record_mask.to(device), record_type_ids.to(device))[0]
        hid_record = hid_record[:, 0, :]
        hid_record = hid_record.cpu().detach()

        if use_layernorm:
            hid_record = LN(hid_record)
        hid_record = torch.cat([hid_record, torch.zeros(max_length_records - hid_record.size(0), model_output_dim)], 0)
        # record_lookup = nn.Embedding.from_pretrained(hid_record, freeze=True).to(device)
        hid_records.append(hid_record)

    for i in range(len(scores)):
        score = scores[i]
        score = [float(s) for s in score]
        score.extend([0]*(max_length_records-len(score)))
        scores[i] = torch.tensor(score, dtype=torch.float)

    return hid_records, scores
