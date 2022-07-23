# coding=utf-8

import argparse
import json
import logging
import time

import numpy as np
import random
from tqdm import tqdm
import os
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset, TensorDataset, RandomSampler
from transformers import AdamW, BertForSequenceClassification, BertTokenizer, BertConfig, \
    get_cosine_schedule_with_warmup
from torch import nn, optim
from transformers.models.bert.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_LIST
from common.dataset_rotowire import RotowireDataset
from common.data_preprocess import prepare_for_bert
from model import Bert_Model

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer)
}
ALL_MODELS = BERT_PRETRAINED_MODEL_ARCHIVE_LIST

logger = logging.getLogger(__name__)


def set_seed(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)


def train_and_eval(model, train_loader, val_loader,
                   optimizer, scheduler, device, epoch, model_dir):
    best_loss = 0.0
    F1score = 0.0
    criterion = nn.MSELoss()
    for i in range(epoch):
        """训练模型"""
        start = time.time()
        model.train()
        print("***** Running training epoch {} *****".format(i + 1))
        train_loss_sum = 0.0
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (ids, att, tpe, record_pos, y) in loop:
            ids, att, tpe, record_pos, y = ids.to(device), att.to(device), tpe.to(device), record_pos.to(device), y.to(
                device)
            y_pred = model(ids, att, tpe, record_pos)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()  # 学习率变化
            train_loss_sum += loss.item()
            loop.set_description(f'Epoch [{i + 1}/{epoch}]')
            loop.set_postfix({'loss': '{:.5f}'.format(loss.item())})
        print('Train | Loss:{:.5f} '.format(train_loss_sum / len(train_loader)))

        # torch.save(model, os.path.join(model_dir, 'ckpt_bert_relu.model'))
        # """验证模型"""
        model.eval()
        val_loss = evaluate(model, val_loader, device)  # 验证模型的性能
        # 保存最优模型
        if val_loss > best_loss:
            best_loss = val_loss
            torch.save(model, os.path.join(model_dir, 'ckpt_bert.model'))
        print("current loss is {:.5f}, best loss is {:.5f}".format(val_loss, best_loss))
        model.train()


def evaluate(model, valid_loader, device):
    tot_loss = 0
    criterion = nn.MSELoss()
    val_pred = []
    for idx, (ids, att, tpe, record_pos) in tqdm(enumerate(valid_loader)):
        output = model(ids.to(device), att.to(device), tpe.to(device), )
        # y_pred = torch.argmax(y_pred, dim=1).detach().cpu().numpy().tolist()
        # val_pred.extend(y_pred)
        loss = criterion()
        tot_loss += loss.item()

    return val_pred, tot_loss / len(valid_loader)


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
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
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

    model = Bert_Model(args.model_name_or_path)
    model.bert.resize_token_embeddings(len(tokenizer))
    model.to(device)
    epoch = args.num_train_epochs

    if args.do_train:
        # create example
        logger.info(f'use model:{model}')
        path_dic = {}
        for example_type in ['train', 'val', 'test']:
            path_dic[example_type] = os.path.join(args.data_dir, f'bert_data/{example_type}_data.pt')
            file = path_dic[example_type]
            if not os.path.exists(file):
                input_ids, input_types, input_masks, y, record_pos = prepare_for_bert(args.data_dir, tokenizer,
                                                                                      example_type=example_type)
                torch.save([input_ids, input_types, input_masks, y, record_pos], file)
                print(f"saving {example_type}_data")
        # 加载数据
        input_ids_train, input_types_train, input_masks_train, y_train, record_pos_train = torch.load(path_dic['train'])
        input_ids_val, input_types_val, input_masks_val, y_val, record_pos_val = torch.load(path_dic['train'])
        # 封装 训练验证数据
        train_data = TensorDataset(input_ids_train, input_masks_train, input_types_train, record_pos_train, y_train)
        val_data = TensorDataset(input_ids_val, input_masks_val, input_types_val, record_pos_val, y_val)
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        val_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=False)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)  # AdamW优化器
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                                    num_training_steps=epoch * len(train_loader))

        logger.info(f"batch={args.train_batch_size},lr = {args.learning_rate} ")
        train_and_eval(model, train_loader, val_loader, optimizer, scheduler, device, epoch,
                       os.path.join(args.data_dir, 'bert_saved'))

    if args.do_eval:
        pass


if __name__ == '__main__':
    main()
