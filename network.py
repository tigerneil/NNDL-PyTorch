# -*- coding: utf-8 -*-
"""通过pytorch中神经网络的模块和函数来构建对MNIST数据集网络的构建、
训练、验证、和测试，整个过程使用了三层的神经元的网络来建立网络；最后
测试集中的正确率有94.36%左右，通过增加网络层数，调整参数，迭代次数，
损失函数等等都能对提高正确率起一定效果

Author: Jing Li
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import mnist_loader


class Network(nn.Module):
    """这里以[784 30 10]三层神经网络为例
    """
    def __init__(self, sizes):
        super(Network, self).__init__()
        self.sizes = sizes
        self.layer1 = nn.Linear(sizes[0], sizes[1])
        self.layer2 = nn.Linear(sizes[1], sizes[2])

    def forward(self, a):
        a = a.view(-1, self.sizes[0])  # view函数将输入Tensor转换成（64, 784）
        a = self.layer1(a)
        a = self.layer2(a)
        a = torch.log_softmax(a, dim=1)
        return a


def rightness(output, target):
    """输入网络的输出Tensor和目标Tensor，
    比较网络的输出Tensor和目标Tensor中对应相等的结果，
    返回比较结果中匹配正确的个数和整个输出或者目标Tensor
    的长度
    """
    rights = 0
    for index in range(len(target.data)):
        if torch.argmax(output[index]) == target.data[index]:
            rights += 1
    return rights, len(target.data)


def train_model(train_loader, epochs, eta):
    """本函数的功能是训练模型，使用交叉熵的损失函数，和
    随机梯度下降的优化算法，学习率为0.001，动量为0.9
    开始训练循环
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=eta, momentum=0.9)

    for epoch in range(epochs):
        train_rights = []  # 记录每次迭代正确的结果和总样本

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            net.train()

            output = net(data)
            loss = criterion(output, target)
            optimizer.zero_grad()  # 清空梯度
            loss.backward()  # 反向传播
            optimizer.step()  # 一步随机梯度下降算法
            right = rightness(output, target)  # 计算一批次准确率中（正确样例数， 总样本数）
            train_rights.append(right)

            if batch_idx % 100 == 0:
                validation_model(validation_loader)

        # 求得整个训练样本中正确的样例总数， 和总样本数，可以通过两者得到训练的正确率
        train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
        print("Epoch {0}: {1}/{2}".format(epoch, train_r[0], train_r[1]))


def validation_model(validation_loader):
    """验证模型
    """
    net.eval()
    val_rights = []

    for data, target in validation_loader:
        data, target = Variable(data), Variable(target)
        output = net(data)
        right = rightness(output, target)
        val_rights.append(right)

    val_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
    print("验证集的正确率为{:.2f}%".format(100.0 * val_r[0] / val_r[1]))


def test_model(test_loader):
    """测试模型
    """
    net.eval()
    vals = []
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = net(data)
        val = rightness(output, target)
        vals.append(val)

    rights = (sum([tup[0] for tup in vals]), sum([tup[1] for tup in vals]))
    print("测试集的正确率为{:.2f}%".format(100.0 * rights[0] / rights[1]))


train_loader, validation_loader, test_loader = mnist_loader.load_data()
net = Network([784, 30, 10])
train_model(train_loader, 20, 0.001)
test_model(test_loader)
