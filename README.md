# Image Classification

## Introduction

本实验是对图像分类的一个简单实现，使用的是CIFAR-10数据集，包含10个类别，每个类别6000张图像，图像大小为32x32，共50000张训练图像和10000张测试图像。

## Models

### VGG

VGG是一个非常简单的模型，由多个卷积层和池化层组成，最后接全连接层，输出10个类别。

```python
class MyVGGNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2d(64, 64, 3, padding=1)

        self.block2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2d(128, 128, 3, padding=1)

        self.block3_conv1 = nn.Conv2d(128, 256, 3, padding=1)
        self.block3_conv2 = nn.Conv2d(256, 256, 3, padding=1)

        self.block4_conv1 = nn.Conv2d(256, 512, 3, padding=1)
        self.block4_conv2 = nn.Conv2d(512, 512, 3, padding=1)

        self.block5_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.block5_conv2 = nn.Conv2d(512, 512, 3, padding=1)

        self.fc1 = nn.Linear(512 * 1 * 1, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.relu(self.block1_conv1(x))
        out = self.relu(self.block1_conv2(out))
        out = self.pool(out)

        out = self.relu(self.block2_conv1(out))
        out = self.relu(self.block2_conv2(out))
        out = self.pool(out)

        out = self.relu(self.block3_conv1(out))
        out = self.relu(self.block3_conv2(out))
        out = self.pool(out)

        out = self.relu(self.block4_conv1(out))
        out = self.relu(self.block4_conv2(out))
        out = self.pool(out)

        out = self.relu(self.block5_conv1(out))
        out = self.relu(self.block5_conv2(out))
        out = self.pool(out)

        out = torch.flatten(out, 1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out
```

这里与原版VGG的区别在于，原版VGG的卷积层更多，这里出于减少计算量和防止过拟合的考虑，减少了3个卷积层。

### ResNet

ResNet是一个残差网络，由多个残差块组成，每个残差块包含两个卷积层，每个卷积层后面接一个BN层，最后接一个ReLU激活函数。每个残差块的输出是输入加上卷积层的输出，这样可以避免梯度消失的问题。

```python
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class MyResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avg_pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
```

和原版的最基础的 Res18 基本相同，没有做什么改动。

## Training

训练过程，对给出的代码进行了一些修改，主要是修改了一些训练过程中的参数，如 batch_size 增加到64，训练轮数增加到10轮，优化器从
SGD 改为 Adam。

## Test

### VGG

**Accuracy of the network on the 10000 test images: 74%**

| Class | Accuracy |
|:-----:|:--------:|
| plane |  0.742   |
|  car  |  0.847   |
| bird  |  0.604   |
|  cat  |  0.528   |
| deer  |  0.661   |
|  dog  |  0.634   |
| frog  |  0.820   |
| horse |  0.863   |
| ship  |  0.896   |
| truck |  0.889   |

### ResNet

**Accuracy of the network on the 10000 test images: 83%**

| Class | Accuracy |
|:-----:|:--------:|
| plane |  0.871   |
|  car  |  0.949   |
| bird  |  0.754   |
|  cat  |  0.699   |
| deer  |  0.859   |
|  dog  |  0.814   |
| frog  |  0.775   |
| horse |  0.797   |
| ship  |  0.916   |
| truck |  0.881   |
