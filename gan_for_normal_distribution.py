#!/usr/bin/env python
# encoding: utf-8
'''
@author: lixiaoshuang
@time: 2019.4.26 19:26
@file: gan_for_normal_distribution.py
@desc:
'''
from pathlib import Path
import os, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt


# 判别器
class disciminator(nn.Module):
    def __init__(self):
        super(disciminator, self).__init__()
        self.fc1 = nn.Linear(SAMPLE_NUM, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.sigmoid(x)


# 生成器
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(SAMPLE_NUM, 512)
        self.fc2 = nn.Linear(512, SAMPLE_NUM)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    plt.ion()  # 开启interactive mode，便于连续plot
    # 用于计算的设备 CPU or GPU
    device = torch.device("cuda" if USE_CUDA else "cpu")
    # 定义判别器与生成器的网络
    net_d = disciminator()
    net_g = generator()
    net_d.to(device)
    net_g.to(device)
    # 损失函数
    criterion = nn.BCELoss().to(device)
    # 真假数据的标签
    true_lable = Variable(torch.ones(BATCH_SIZE,1)).to(device)
    fake_lable = Variable(torch.zeros(BATCH_SIZE,1)).to(device)
    # 优化器
    optimizer_d = torch.optim.Adam(net_d.parameters(), lr=0.0005)
    optimizer_g = torch.optim.Adam(net_g.parameters(), lr=0.0005)

    for i in range(MAX_EPOCH):
        # 生成生态分布真实数据
        real_data = np.vstack([np.random.normal(0, 0.05, SAMPLE_NUM) for _ in range(BATCH_SIZE)])
        real_data = Variable(torch.Tensor(real_data)).to(device)
        # 用均匀分布噪声作为生成器的输入
        g_noises = np.array([np.random.uniform(0.1,4,SAMPLE_NUM) for _ in range(BATCH_SIZE)])
        g_noises = Variable(torch.Tensor(g_noises)).to(device)
        # 训练辨别器
        optimizer_d.zero_grad()
        # 辨别器辨别真图的loss
        d_real = net_d(real_data)
        loss_d_real = criterion(d_real, true_lable)
        loss_d_real.backward()
        # 辨别器辨别假图的loss
        fake_date = net_g(g_noises)
        d_fake = net_d(fake_date)
        loss_d_fake = criterion(d_fake, fake_lable)
        loss_d_fake.backward()
        optimizer_d.step()
        # 训练生成器

        optimizer_g.zero_grad()
        fake_date = net_g(g_noises)
        d_fake = net_d(fake_date)
        # 生成器生成假图的loss
        loss_g = criterion(d_fake, true_lable)
        loss_g.backward()
        optimizer_g.step()
        # 每200步画出生成的数字图片和相关的数据
        if i % 200 == 0:
            # print('fake_date[0]:  \n', fake_date[0])
            plt.cla()
            plt.hist(fake_date[0].to('cpu').detach().numpy(),25, color='blue',alpha=0.5,label=['gen','true'])  # 生成网络生成的数据
            plt.hist(real_data[0].to('cpu').detach().numpy(), 25,color='black',alpha=0.5,label=['true','gen'])  # 真实数据
            plt.legend()
            prob = (loss_d_real.mean() + 1 - loss_d_fake.mean()) / 2.
            plt.title('D accuracy=%.2f (0.5 for D to converge)' % (prob),
                     fontdict={'size': 15})
            plt.ylim(0, 120)
            plt.xlim(-0.25, 0.25)
            #             plt.savefig(Path(os.getcwd()) / 'results' / 'sin_wave' / Path(str(i) + '_epoch_plot.png'), dpi=300)
            plt.draw(), plt.pause(0.2)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    SAMPLE_NUM = 1000
    BATCH_SIZE = 64
    USE_CUDA = True
    MAX_EPOCH = 100000

    main()