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
from common.data_preprocess import prepare_for_bert
from model import Bert_Model
from common.metric import get_f1score_transformer, get_acc_transformer, get_recall_transformer, \
    get_precision_transformer

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
    best_loss = 1.0
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
        _, _, val_loss, _, _, _ = evaluate(model, val_loader, device)  # 验证模型的性能
        # 保存最优模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(model_dir, 'ckpt_bert_norm.model')
            torch.save(model, save_path)
            print('saving model in %s' % save_path)
        print("current loss is {:.5f}, best loss is {:.5f}".format(val_loss, best_loss))
        model.train()


def evaluate(model, valid_loader, device, tab_lens=None):
    tot_loss = 0
    label_list = []
    pred_list = []
    criterion = nn.MSELoss()
    val_pred = []
    val_label = []
    with torch.no_grad():
        if not tab_lens:
            for idx, (ids, att, tpe, record_pos, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                output = model(ids.to(device), att.to(device), tpe.to(device), record_pos.to(device))
                pred = _norm(output) > 0.5
                ll = _norm(labels) > 0.5
                label_list.extend(ll.view(-1).cpu().numpy().astype(int))
                pred_list.extend(pred.view(-1).cpu().numpy().astype(int))
                loss = criterion(labels.to(device), output)
                # 统计比例
                val_pred.extend(pred.view(-1, 20).cpu().numpy().astype(int))
                val_label.extend(ll.view(-1, 20).cpu().numpy().astype(int))
                tot_loss += loss.item()
            f1, acc, recall, precision = get_f1score_transformer(label_list, pred_list), get_acc_transformer(label_list,
                                                                                                             pred_list), get_recall_transformer(
                label_list, pred_list), get_precision_transformer(label_list, pred_list)
            print(
                "test | Loss:{:.5f}  f1_score:{:.5f} accuracy:{:.5f} recall:{:.5f} precision:{:.5f}".format(
                    tot_loss / len(valid_loader), f1, acc, recall, precision))
        else:
            n = 0
            count = 0
            pred = []
            ll = []
            for idx, (ids, att, tpe, record_pos, labels) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
                output = model(ids.to(device), att.to(device), tpe.to(device), record_pos.to(device))
                loss = criterion(labels[0].to(device), output)
                tot_loss += loss.item()
                # pred.append(output.cpu().numpy())
                # ll.append(labels[0].cpu().numpy())
                pred.extend(output.cpu().tolist())
                ll.extend(labels[0].cpu().tolist())
                count += 1
                if count == tab_lens[n]:
                    # pred = _norm(pred) > 0.5
                    # ll = np.array(ll) > 0.5
                    pred = set_top(pred, k=80)
                    ll = set_top(ll, k=80)
                    label_list.extend(ll)
                    pred_list.extend(pred)
                    # label_list.extend(ll.reshape(-1).astype(int))
                    # pred_list.extend(pred.reshape(-1).astype(int))
                    # reset变量
                    pred = []
                    ll = []
                    n += 1
                    count = 0

            f1, acc, recall, precision = get_f1score_transformer(label_list, pred_list), get_acc_transformer(label_list,
                                                                                                             pred_list), get_recall_transformer(
                label_list, pred_list), get_precision_transformer(label_list, pred_list)

            print("test | Loss:{:.5f}  f1_score:{:.5f} accuracy:{:.5f} recall:{:.5f} precision {:.5f}".format(
                tot_loss / len(valid_loader), f1,
                acc, recall, precision))

    return val_pred, val_label, tot_loss / len(valid_loader), f1, acc, recall


def _norm(a):
    return (a - torch.min(a, dim=1).values.view(-1, 1)) / (
            torch.max(a, dim=1).values.view(-1, 1) - torch.min(a, dim=1).values.view(-1, 1))


# def _norm(a):
#     return (a - np.min(a)) / (np.max(a) - np.min(a))


def set_top(r, k=75):
    tmp = sorted(r, reverse=True)[k]
    return [0 if rr < tmp else 1 for rr in r]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="Task database.")
    parser.add_argument("--data_src", default=None, type=str, required=True,
                        help="source database.")
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
    parser.add_argument("--num_train_epochs", default=5, type=int,
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

    epoch = args.num_train_epochs
    # create example
    path_dic = {}
    for example_type in ['train', 'valid', 'test']:
        path_dic[example_type] = os.path.join(args.data_dir, f'{example_type}_data.pt')
        file = path_dic[example_type]
        if not os.path.exists(file):
            input_ids, input_types, input_masks, y, record_pos, tab_lens = prepare_for_bert(args.data_src,
                                                                                            args.data_dir, tokenizer,
                                                                                            example_type=example_type,
                                                                                            do_norm=False)
            torch.save([input_ids, input_types, input_masks, y, record_pos, tab_lens], file)
            print(f"saving {example_type}_data")

    if args.do_train:
        # 加载模型
        model = Bert_Model(args.model_name_or_path)
        model.bert.resize_token_embeddings(len(tokenizer))
        model.to(device)
        logger.info(f'use model for training:{model}')
        # 加载数据
        input_ids_train, input_types_train, input_masks_train, y_train, record_pos_train, _ = torch.load(
            path_dic['train'])
        input_ids_val, input_types_val, input_masks_val, y_val, record_pos_val, _ = torch.load(path_dic['valid'])
        # 封装 训练验证数据
        train_data = TensorDataset(input_ids_train, input_masks_train, input_types_train, record_pos_train, y_train)
        val_data = TensorDataset(input_ids_val, input_masks_val, input_types_val, record_pos_val, y_val)
        train_sampler = RandomSampler(train_data)
        train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        val_loader = DataLoader(val_data, batch_size=args.train_batch_size, shuffle=False)
        optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)  # AdamW优化器
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=len(train_loader),
                                                    num_training_steps=epoch * len(train_loader))

        logger.info(f"batch={args.train_batch_size},lr = {args.learning_rate} ")
        train_and_eval(model, train_loader, val_loader, optimizer, scheduler, device, epoch,
                       '/Myhome/slf/work/data-to-text/data/bert_saved')

    if args.do_eval:
        # evaluate
        model = torch.load('/Myhome/slf/work/data-to-text/data/bert_saved/ckpt_bert_norm.model')
        model.to(device)
        logger.info(f'use model for evaluating:{model}')
        input_ids_test, input_types_test, input_masks_test, y_test, record_pos_test, tab_lens = torch.load(
            path_dic['test'])
        test_data = TensorDataset(input_ids_test, input_masks_test, input_types_test, record_pos_test, y_test)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
        test_pred, test_label, _, _, _, _ = evaluate(model, test_loader, device, tab_lens=None)
        print(1)

if __name__ == '__main__':
    main()
