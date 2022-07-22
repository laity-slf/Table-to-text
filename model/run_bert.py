# coding=utf-8

import argparse
import json
import logging
import numpy as np
import random
from tqdm import tqdm
import os
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, BertConfig
from torch import nn, optim
from transformers.models.bert.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_LIST
from common.utils import create_examples
from common.lookup import get_record_lookup_from_first_token
from common.dataset_rotowire import RotowireDataset
from common.data_preprocess import prepare_for_bert
from model import RecordEncoding, Regression

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}
ALL_MODELS = BERT_PRETRAINED_MODEL_ARCHIVE_LIST

logger = logging.getLogger(__name__)


def set_seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Task database.")
    # parser.add_argument("--predict_type", default=None, type=str, required=True,
    #                     help="Portion of the data to perform prediction on (e.g., dev, test).")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    # Other parameters
    parser.add_argument("--config_name", default="../data/bert", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="Maximum input length after tokenization. Longer sequences will be truncated, shorter ones padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the <predict_type> set.")
    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()
    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    # 随机数
    set_seed(2022)
    # args.model_type
    config_class, tokenizer_class = BertConfig, BertTokenizer
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=True)

    # config 参数调整
    # config.num_labels = 1  # 回归任务

    # create example
    train_data, train_label = prepare_for_bert(args.data_dir, tokenizer, example_type='train')
    val_data, val_label = prepare_for_bert(args.data_dir, example_type='valid')


if __name__ == '__main__':
    main()
