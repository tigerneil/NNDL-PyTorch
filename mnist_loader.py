# -*- coding: utf-8 -*-
"""
Author: Jing Li
"""


import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms

batch_size = 64


def load_data():
    """通过调用torchvision中datasets模块来导入MNIST数据集中的训练集和测试集，
    将导入的训练集通过DataLoader加载为train_loader，
    测试集总共有10，000的样本数，分别取5，000作为验证集和测试集，对验证机和测试
    集中样本打乱通过DataLoader加载为validation_loader和test_loader
    """
    train_dataset = dsets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = dsets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    indices = range(len(test_dataset))
    indices_val = indices[:5000]
    indices_test = indices[5000:]

    sampler_val = torch.utils.data.sampler.SubsetRandomSampler(indices_val)
    sampler_test = torch.utils.data.sampler.SubsetRandomSampler(indices_test)

    validation_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                                    sampler=sampler_val)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                                              sampler=sampler_test)

    return train_loader, validation_loader, test_loader

