from time import strftime

import torch as t
import os
import torchvision as tv
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
from torch.optim import Adam
import tqdm


import models
from attack.PGDAttack import LinfPGDAttack
from config import Config
DOWNLOAD_CIFAR = False #是否下载数据集



def train():
    '''
    训练神经网络
    :return:
    '''
    #1.加载配置
    opt = Config()
    opt._parese()
    global DOWNLOAD_CIFAR

    #1a.加载模型
    model = getattr(models,opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device)

    #2.定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=opt.lr)


    #3.加载数据
    if not (os.path.exists('./data/CIFAR/')) or not os.listdir('./data/CIFAR/'):
        DOWNLOAD_CIFAR =True

    transform = tv.transforms.Compose([
        tv.transforms.Resize(224),
        #要先完成所有操作，然后totensor，normalize
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=opt.MEAN, std=opt.STD)
    ])
    train_data = tv.datasets.CIFAR10(
        root='./data/CIFAR/',
        train=True,
        transform=transform,
        download=DOWNLOAD_CIFAR
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
                              )

    test_data = tv.datasets.CIFAR10(
        root='./data/CIFAR/',
        train = False,
        transform=transform,
        download=DOWNLOAD_CIFAR
    )
    # test_data.loader
    test_loader = DataLoader(test_data, batch_size=opt.test_num, shuffle=True, num_workers=opt.num_workers)

    #训练模型
    for epoch in range(opt.train_epoch):
        for ii,(data,label) in tqdm.tqdm(enumerate(train_loader)):
            input = data.to(opt.device)
            target = label.to(opt.device)

            optimizer.zero_grad()
            score = model(input)
            loss = criterion(score,target)
            loss.backward()
            optimizer.step()

            if (ii+1)%opt.print_freq ==0:
                print('loss:%.2f'%loss.cpu().data.numpy())
        if (epoch + 1) % opt.save_every == 0:
            model.save(epoch=epoch+1)

        #测试
        for ii, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(opt.device)
            test_score = model(test_x)
            correct = np.sum((t.argmax(test_score.to('cpu'), 1) == test_y).numpy())
            print('accuracy: {} '.format(round(correct/test_y.size(0),4)))
            break

@t.no_grad()
def test_acc():
    #测试准确率
    # 1. 加载配置
    opt = Config()
    opt._parese()
    global DOWNLOAD_CIFAR
    if not (os.path.exists('./data/CIFAR/')) or not os.listdir('./data/CIFAR/'):
        DOWNLOAD_CIFAR=True

    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device)
    model.eval()

    # 2.加载数据
    transform = tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=opt.MEAN, std=opt.STD)

    ])

    test_data = tv.datasets.CIFAR10(
        root='./data/CIFAR/',
        train = False,
        transform=transform,
        download=DOWNLOAD_CIFAR
    )
    #test_data.loader
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=opt.num_workers)

    correct_num = 0
    for ii, (test_x, test_y) in enumerate(test_loader):
        if ii >= opt.test_num:
            break
        test_x = test_x.to(opt.device)
        test_score = model(test_x)
        correct = (t.argmax(test_score.to('cpu'), 1) == test_y).numpy()
        correct_num = correct_num + correct[0]

    accuracy = correct_num/opt.test_num
    print('test accuracy:%.2f' % accuracy)
    return accuracy


def train_adv_PGD():
    '''
    使用PGD的生成样本重新开始进行随机初始化训练
    :return:
    '''

    # 1.加载配置
    opt = Config()
    opt._parese()
    global DOWNLOAD_CIFAR
    if not (os.path.exists('./data/CIFAR/')) or not os.listdir('./data/CIFAR/'):
        DOWNLOAD_CIFAR=True

    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device).eval()

    #2.定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=opt.lr)

    # 3.加载数据
    transform = tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=opt.MEAN,std= opt.STD)
    ])

    train_data = tv.datasets.CIFAR10(
        root='./data/CIFAR/',
        train=True,
        transform=transform,
        download=DOWNLOAD_CIFAR
    )

    train_loader = DataLoader(
        dataset=train_data,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers
    )
    test_data = tv.datasets.CIFAR10(
        root='./data/CIFAR/',
        train = False,
        transform=transform,
        download=DOWNLOAD_CIFAR
    )
    # test_data.loader
    test_loader = DataLoader(test_data, batch_size=opt.test_num, shuffle=True, num_workers=opt.num_workers)

    attack = LinfPGDAttack(model,epsilon=opt.epsilon)
    # 训练模型
    for epoch in range(opt.train_epoch):
        for ii, (data, label) in tqdm.tqdm(enumerate(train_loader)):
            optimizer.zero_grad()

            perturb_x = attack.perturb(data.numpy(), label)

            score = model(t.FloatTensor(perturb_x).to(opt.device))
            loss = criterion(score, label)
            loss.backward()
            optimizer.step()

            if (ii + 1) % opt.print_freq == 0:
                print('loss:%.2f' % loss.cpu().data.numpy())
        if (epoch + 1) % opt.save_every == 0:
            model.save(epoch=epoch + 1,isAdv=True)

        #测试
        for ii, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(opt.device)
            test_score = model(test_x)
            correct = np.sum((t.argmax(test_score.to('cpu'), 1) == test_y).numpy())
            print('accuracy: {} '.format(round(correct/test_y.size(0),4)))
            break

def attack_PGD():
    '''
    对使用真实样本和对抗样本训练的模型进行攻击
    :return:
    '''
    # 1.加载配置
    opt = Config()
    opt._parese()
    global DOWNLOAD_CIFAR
    if not (os.path.exists('./data/CIFAR/')) or not os.listdir('./data/CIFAR/'):
        DOWNLOAD_CIFAR = True
    # 1a.加载模型
    model = getattr(models, opt.model)()
    if opt.model_path:
        model.load(opt.model_path)
    model.to(opt.device).eval()
    # 2.加载数据
    transform = tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=opt.MEAN, std=opt.STD)
    ])
    test_data = tv.datasets.CIFAR10(
        root='./data/CIFAR/',
        train=False,
        transform=transform,
        download=DOWNLOAD_CIFAR
    )
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)
    success_num = 0
    attack = LinfPGDAttack(model)
    for ii, (data, label) in enumerate(test_loader):
        if ii >= opt.test_num:
            break
        test_score = model(data.to(opt.device))
        if t.argmax(test_score.to('cpu'), 1) == label.to('cpu'):
            continue
        perturb_x = attack.perturb(data.numpy(), label)
        test_score = model(t.FloatTensor(perturb_x).to(opt.device))
        if t.argmax(test_score.to('cpu'), 1) != label:
            success_num += 1
    success_rate = success_num / ii
    accuracy = test_acc()
    accuracy_after = accuracy * (1 - success_rate)
    string = '{} , {} , {} , {} \n'.format(opt.model_path, accuracy, accuracy_after, success_rate)
    open('log.csv', 'a').write(string)


if __name__ == '__main__':
   train()