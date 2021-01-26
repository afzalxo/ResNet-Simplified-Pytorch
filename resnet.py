import torch
import torch.nn as nn
from typing import Type, List, Union

#Simplifying Torch implementation of resnet.
#Adopted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

def conv3x3(in_planes: int, out_planes: int, stride: int=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int=1
    def __init__(self, in_depth, out_depth, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_depth, out_depth, stride=stride)
        self.conv2 = conv3x3(out_depth, out_depth, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_depth)
        self.bn2 = nn.BatchNorm2d(out_depth)
        self.downsample=downsample

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class BottleneckBlock(nn.Module):
    expansion: int = 4
    def __init__(self, in_depth, neck_depth, stride=1, downsample=None):
        super(BottleneckBlock, self).__init__()
        self.conv1 = conv1x1(in_depth, neck_depth, stride = 1)
        self.conv2 = conv3x3(neck_depth, neck_depth, stride=stride)
        self.conv3 = conv1x1(neck_depth, neck_depth * self.expansion, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(neck_depth)
        self.bn2 = nn.BatchNorm2d(neck_depth)
        self.bn3 = nn.BatchNorm2d(neck_depth * self.expansion)
        self.downsample = downsample

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block: Type[Union[BasicBlock, BottleneckBlock]], layers: List[int], num_classes: int = 10):
        super(ResNet, self).__init__()
        self.in_depth = 64

        self.block0 = nn.Sequential(
                nn.Conv2d(3, self.in_depth, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(self.in_depth),
                nn.ReLU(inplace=True),
                #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
        self.block1 = self._make_layer(block, 64, layers[0]) 
        self.block2 = self._make_layer(block, 128, layers[1], stride=2) 
        self.block3 = self._make_layer(block, 256, layers[2], stride=2) 
        self.block4 = self._make_layer(block, 512, layers[3], stride=2) 
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') #He init for conv weights
            ##No need to initialize BN weights biases manually since they initialize to 1 and 0, respectively, by default.
        '''

    def _make_layer(self, block: Type[Union[BasicBlock, BottleneckBlock]], neck_depth: int, blocks: int, stride: int=1):
        downsample = None
        if stride != 1 or self.in_depth != neck_depth * block.expansion:
            downsample = nn.Sequential(
                    conv1x1(self.in_depth, neck_depth*block.expansion, stride),
                    nn.BatchNorm2d(neck_depth*block.expansion)
                    )
        layers = []

        layers.append(block(self.in_depth, neck_depth, stride, downsample))
        self.in_depth = neck_depth*block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_depth, neck_depth))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18():
    return ResNet(BasicBlock, [2,2,2,2])

def resnet34():
    return ResNet(BasicBlock, [3,4,6,3])

def resnet50():
    return ResNet(BottleneckBlock, [3,4,6,3])

def resnet101():
    return ResNet(BottleneckBlock, [3,4,23,3])

def resnet152():
    return ResNet(BottleneckBlock, [3,8,36,3])

