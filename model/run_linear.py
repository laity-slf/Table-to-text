import os

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common.dataset_rotowire import RotowireDataset
from common.metric import get_f1score_transformer, get_acc_transformer, get_precision_transformer, \
    get_recall_transformer
from model import linear_model


def training(n_epoch, lr, model_dir, train, valid, model, device):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))
    model.train()
    criterion = nn.BCELoss()
    t_batch = len(train)
    v_batch = len(valid)
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    total_loss, total_acc, best = 0, 0, 0
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
        # with torch.no_grad():
        #     total_loss = 0
        #     for i, (inputs, labels) in enumerate(valid):
        #         inputs = inputs.to(device)
        #         labels = labels.to(device, dtype=torch.float)
        #         outputs = model(inputs)
        #         outputs = outputs.squeeze()
        #         loss = criterion(outputs, labels)
        #         total_loss += loss.item()
        total_loss, f1, accuracy, recall, precision = evaluate(model, valid, device)

        # print("Valid | Loss:{:.5f}".format(total_loss / v_batch))
        if f1 > best:
            best = f1
            # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
            torch.save(model, os.path.join(model_dir, f'ckpt_f1_{f1}.model'))
            print('saving model with f1 {:.5f}'.format(f1, best))
        print(f"best f1: {best:.5f}")
    print('-----------------------------------------------')
    model.train()


def evaluate(model, valid_loader, device, tab_lens=None):
    tot_loss = 0
    label_list = []
    pred_list = []
    criterion = nn.BCELoss()
    # val_pred = []
    # val_label = []
    for i, (inputs, labels) in enumerate(valid_loader):
        inputs = inputs.to(device)
        labels = labels.to(device, dtype=torch.float)
        outputs = model(inputs)
        outputs = outputs.squeeze()
        loss = criterion(outputs, labels)
        label_list.extend(labels.tolist())
        pred_list.extend(outputs.ge(0.5).to(dtype=int).tolist())
        tot_loss += loss.item()

    f1, acc, recall, precision = get_f1score_transformer(label_list, pred_list), get_acc_transformer(label_list,
                                                                                                     pred_list), get_recall_transformer(
        label_list, pred_list), get_precision_transformer(label_list, pred_list)

    print("evaluate | Loss:{:.5f}  f1_score:{:.5f} accuracy:{:.5f} recall:{:.5f} precision {:.5f}".format(
        tot_loss / len(valid_loader), f1,
        acc, recall, precision))

    return tot_loss / len(valid_loader), f1, acc, recall, precision


def convert_data(x):
    def convert(a):
        return list(map(float, a))

    # def convert_y(b):
    #     tmp = b.replace("'", '')[1:-1].split(',')
    #     return list(map(float, tmp))

    x = list(map(convert, x))
    # y = list(map(convert_y, y))
    return torch.tensor(x)


def convert_label(y):
    def convert(a):
        return list(map(float, a))

    # def convert_y(b):
    #     tmp = b.replace("'", '')[1:-1].split(',')
    #     return list(map(float, tmp))

    y = list(map(convert, y))
    label = np.array(np.array(y) > 0.5).astype(int)
    _y = []
    for xx in label:
        _y.append(1 if 1 in xx else 0)
    # y = list(map(convert_y, y))
    return torch.tensor(_y)


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def add_all_star(df, dic):
    all_star = []
    for n in df['Name'].values:
        all_star.append(dic[n])
    df['all_star'] = all_star
    return df


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_prefix = '../data/'
    # 初次筛选出来的行
    row_train = np.load('../data/pred_row_recall_95%_train.npy')
    row_val = np.load('../data/pred_row_recall_95%_valid.npy')
    row_test = np.load('../data/pred_row_recall_95%_test.npy')
    # per
    per_train = np.load(os.path.join(data_prefix, 'per_train.npy'))
    per_valid = np.load(os.path.join(data_prefix, 'per_valid.npy'))
    per_test = np.load(os.path.join(data_prefix, 'per_test.npy'))

    per_train = norm(per_train.reshape(-1, 22)[:, 0])
    per_valid = norm(per_valid.reshape(-1, 22)[:, 0])
    per_test = norm(per_test.reshape(-1, 22)[:, 0])

    # 归一化的data
    train_data = pd.read_csv(os.path.join(data_prefix, 'rotowire_train.csv'))
    val_data = pd.read_csv(os.path.join(data_prefix, 'rotowire_valid.csv'))
    test_data = pd.read_csv(os.path.join(data_prefix, 'rotowire_test.csv'))
    train_data['per'], val_data['per'], test_data['per'] = per_train, per_valid, per_test

    # add all_star
    all_star_train = np.load('../data/all_star_train.npy', allow_pickle=True).item()
    all_star_val = np.load('../data/all_star_valid.npy', allow_pickle=True).item()
    all_star_test = np.load('../data/all_star_test.npy', allow_pickle=True).item()

    train_data = add_all_star(train_data, all_star_train)
    val_data = add_all_star(val_data, all_star_val)
    test_data = add_all_star(test_data, all_star_test)

    train_x, val_x = convert_data(
        train_data.values[row_train.astype(bool), :][:, list(range(3, 8)) + [17, 18, 21, 23, 24]]), convert_data(
        val_data.values[row_val.astype(bool), :][:, list(range(1, 8)) + [17, 18, 21, 23, 24]])
    train_y, val_y = np.load(os.path.join(data_prefix, 'train.npy')), np.load(os.path.join(data_prefix, 'valid.npy'))
    train_y, val_y = convert_label(train_y[row_train.astype(bool)]), convert_label(val_y)
    # 参数
    bs = 64
    n_epoch = 50
    lr = 1e-4

    # 打包数据
    train_data = RotowireDataset(train_x, train_y)
    val_data = RotowireDataset(val_x, val_y)
    train_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=False)

    # 加载模型
    model = linear_model(10, 1)
    model.to(device)

    # 训练
    # training(n_epoch, lr, '/Myhome/slf/work/data-to-text/data/linear', train_loader, val_loader, model, device)

    # 评估
    evaluate(model, )

    print(1)
