#用以测试模型是否能够跑通
from models import MobileNetV2_x2,MobileNetV2,MobileNetV2_x4,ShuffleNetV2_x4,ShuffleNetV2,ShuffleNetV2_x2


import os
from torch import nn
from torch.optim import Adam
import torchvision as tv
from torch.utils.data import DataLoader
DOWNLOAD_CIFAR = False

#1.读取模型
model = ShuffleNetV2_x2()

# 2.定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

if not (os.path.exists('../data/CIFAR/')) or not os.listdir('../data/CIFAR/'):
    DOWNLOAD_CIFAR = True

transform = tv.transforms.Compose([
    # 要先完成所有操作，然后totensor，normalize
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),std=(0.2023, 0.1994, 0.2010))
])
train_data = tv.datasets.CIFAR10(
        root='../data/CIFAR/',
        transform=transform,
        download=DOWNLOAD_CIFAR
    )

dl = DataLoader( dataset=train_data,
        batch_size=4,
        shuffle=True,
        num_workers=0)
for ii, (data, label) in enumerate(dl):


    optimizer.zero_grad()
    score = model(data)
    loss = criterion(score, label)
    loss.backward()
    optimizer.step()

    break
