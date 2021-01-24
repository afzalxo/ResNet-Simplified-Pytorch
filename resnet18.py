import torch
import torch.nn as nn

def conv3x3(in_planes: int, out_planes: int, stride: int=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes: int, out_planes: int, stride: int=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int=1
    def __init__(self, in_ch, out_ch, stride=1 downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.conv2 = conv3x3(out_ch, out_ch, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
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
    def __init__(self, in_res, in_ch, stride=1):
        super(BottleneckBlock, self).__init__()
        neck_ch = 64
        self.conv1 = conv1x1(in_ch, neck_ch, stride = 1)
        self.conv2 = conv3x3(neck_ch, neck_ch, stride=stride)
        self.conv3 = conv1x1(neck_ch, in_ch*2, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(neck_ch)
        self.bn2 = nn.BatchNorm2d(neck_ch)
        self.bn3 = nn.BatchNorm2d(in_ch*2)

    def forward(self,x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x += residual
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, layers, num_classes):
        super().__init__()

        self.block0 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                )
        self.block1 = BottleneckBlock() 
        self.block2 = BottleneckBlock() 
        self.block3 = BottleneckBlock() 
        self.block4 = BottleneckBlock() 

