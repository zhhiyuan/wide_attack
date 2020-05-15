# wide attack
adversarial attack on CIFAR10 to test different wide network
在CIFAR10数据集上测试网络宽度对对抗攻击的影响

### 攻击方法(attack mode)

- PGD attack
code from [there](https://github.com/wanglouis49/pytorch-adversarial_box)
epsilon=0.3, k=40, a=0.01

### 目前模型(model)

- ShuffleNetV2, ShuffleNetV2_x2, ShuffleNetV2_x4

- MobileNetV2, MobileNetV2_x2,MobileNetV2_x4

模型来自`torchvision.model`，更多细节[参考](https://pytorch.org/docs/stable/torchvision/models.html)

不同宽度的模型是自己修改的，可能会影响到准确率

### 环境

- torch==1.1.0

- torchvision==0.3.0

- pillow<7.0.0

- tqdm

### 运行须知

- 运行`main.py`即可，此方法需要配置`config.py`文件。或使用notebook运行`main.ipynb`，此方法不需要配置文件，在ipynb里面配置即可

- 若`./ckps`文件夹下无预训练模型，则需要先训练模型

- 分为train()和train_adv_PGD()函数，train()作用是使用真实数据集训练并将模型保存在ckps文件夹下；train_adv_PGD()用于读取预训练模型并生成对抗样本，样本放入相同的新的随机初始化模型上训练

- 训练完成后测试其对抗性，使用attack_PGD()函数，预训练模型放入config中的model_path中

- 目前只测试过notebook版本