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


def create_examples_and_lookup(args, r_encoder, tokenizer, device, example_type='train'):
    records, scores = create_examples(args.data_dir, example_type=example_type)
    logger.info(f"create {example_type}")
    record_lookup, label = get_record_lookup_from_first_token(records, r_encoder, scores, tokenizer, device)
    return record_lookup, label


def training(n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    criterion = nn.MSELoss()
    t_batch = len(train)
    v_batch = len(valid)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best_loss = 0, 0, 100
    for epoch in range(n_epoch):
        loop = tqdm(enumerate(train), total=t_batch)
        total_loss, total_acc = 0, 0
        # 做training
        for i, (inputs, labels) in loop:
            inputs = inputs.to(device)
            labels = labels.to(device, dtype=torch.float)
            optimizer.zero_grad()
            outputs = model(inputs)
            # 去掉最外面的dimension
            outputs = outputs.squeeze()
            # 计算training loss
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_description(f'Epoch [{epoch + 1}/{n_epoch}]')
            loop.set_postfix({'loss': '{:.5f}'.format(loss.item())})
            # print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.format(
            #     epoch + 1, i + 1, t_batch, loss.item(), correct * 100 / batch_size), end='\r')
            # if (i+1)%100==0:
            #     print('[ Epoch{}: {}/{} ] loss:{:.3f} '.format(
            #         epoch + 1, i + 1, t_batch, loss.item(), end='\r'))
        print('\nTrain | Loss:{:.5f} '.format(total_loss / t_batch))
        # validation
        model.eval()
        with torch.no_grad():
            total_loss = 0
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device)
                labels = labels.to(device, dtype=torch.float)
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item()

            print("Valid | Loss:{:.5f}".format(total_loss / v_batch))
            if total_loss < best_loss:
                best_loss = total_loss
                # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
                torch.save(model, os.path.join(model_dir, 'ckpt.model'))
                print('saving model with loss {:.5f}'.format(total_loss / v_batch))
        print('-----------------------------------------------')
        model.train()


def testing(test_loader, model, device):
    model.eval()
    t_batch = len(test_loader)
    ret_output = []
    criterion = nn.MSELoss()
    total_loss = 0
    label_list = []
    pred_list = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            outputs = outputs.squeeze()
            loss = criterion(outputs, labels)

            total_loss += loss.item()

    print("test | Loss:{:.5f}".format(total_loss / t_batch))
    return ret_output


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
    parser.add_argument("--config_name", default="", type=str,
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
    # args.model_type = args.model_type.lower()
    # 加载配置文件
    config_class, tokenizer_class = BertConfig, BertTokenizer
    config = BertConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                              do_lower_case=True)

    # config 参数调整
    # config.num_labels = 1  # 回归任务
    # encoder
    r_encoder = RecordEncoding.from_pretrained(args.model_name_or_path, config=config)

    logger.info("Updated model config: %s" % config)
    # fix bert
    for p in r_encoder.parameters():
        p.requires_grad = False
    # r_encoder.to(device)

    # create example
    for f in ['train', 'valid', 'test']:
        if not os.path.exists(f'../data/cache_{f}_data_label.pt'):
            record_lookup, label = create_examples_and_lookup(args, r_encoder, tokenizer, device, example_type=f)
            torch.save([record_lookup, label], f'../data/cache_{f}_data_label.pt')
    # load data
    train_data, train_label = torch.load(f'../data/cache_train_data_label.pt')
    val_data, val_label = torch.load(f'../data/cache_valid_data_label.pt')

    train_dataset = RotowireDataset(X=train_data, y=train_label)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=8)

    val_dataset = RotowireDataset(X=val_data, y=val_label)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=8)

    model = Regression(d_model=train_data[0].size(1), num_layers=6)
    model.to(device)
    epoch = 10
    lr = 1e-6
    if args.do_train:
        logger.info("Training parameters %s", args)
        training(epoch, lr, '../data', train_loader, val_loader, model, device)

    if args.do_eval:
        model = torch.load('../data/ckpt.model')
        model.to(device)
        test_data, test_label = torch.load(f'../data/cache_test_data_label.pt')
        test_dataset = RotowireDataset(X=test_data, y=test_label)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.train_batch_size, shuffle=False, num_workers=8)

        res = testing(test_loader, model, device)
        np.save('../data/res.npy', res)


if __name__ == "__main__":
    main()
