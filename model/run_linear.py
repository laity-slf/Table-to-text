import os

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch import optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common.dataset_rotowire import RotowireDataset
from model import linear_model


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


def evaluate():
    pass


def convert_data(x):
    def convert(a):
        return list(map(float, a))

    # def convert_y(b):
    #     tmp = b.replace("'", '')[1:-1].split(',')
    #     return list(map(float, tmp))

    x = list(map(convert, x))
    # y = list(map(convert_y, y))
    return torch.tensor(x)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_prefix = '../data/'
    train_data = pd.read_csv(os.path.join(data_prefix, 'rotowire_train.csv'))
    val_data = pd.read_csv(os.path.join(data_prefix, 'rotowire_valid.csv'))

    train_x, val_x = convert_data(train_data.values[:, 1:]), convert_data(val_data.values[:, 1:])
    train_y, val_y = np.load(os.path.join(data_prefix, 'train.npy')), np.load(os.path.join(data_prefix, 'valid.npy'))
    # 参数
    bs = 64
    n_epoch = 10
    lr = 1e-4

    # 打包数据
    train_data = RotowireDataset(train_x, torch.from_numpy(train_y))
    val_data = RotowireDataset(val_x, torch.from_numpy(val_y))
    train_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=True)
    val_loader = DataLoader(dataset=train_data, batch_size=bs, shuffle=False)

    # 加载模型
    model = linear_model(22, 20)
    model.to(device)
    training(n_epoch, lr, '/Myhome/slf/work/data-to-text/data/linear', train_loader, val_loader, model, device)

    print(1)
