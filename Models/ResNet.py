"""resnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion,affine=True,track_running_stats=True)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion,affine=True,track_running_stats=True)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, counter=0, L=0, t=0, prev_L=0):
        #residual = self.residual_function(x)
        out = x
        for i in range(len(self.residual_function)):
            #print(self.residual_function[i])
            if 'SPIKE_layer' in str(self.residual_function[i]):
                if prev_L <= counter:
                    out = self.residual_function[i](out, t)
            else:
                if prev_L <= counter:
                    out = self.residual_function[i](out)
            if 'SPIKE_layer' in str(self.residual_function[i]):
                counter += 1
                if counter == L:
                    return out, counter, True
            if 'ReLU' in str(self.residual_function[i]):
                counter += 1
                if counter == L:
                    return out, counter, True
        if prev_L <= counter:
            shortcut = self.shortcut(x)

        if 'SPIKE_layer' in str(self.relu):
            if prev_L <= counter:
                out = self.relu(out + shortcut, t)
            counter += 1
            if counter == L:
                return out, counter, True
        if 'ReLU' in str(self.relu):
            if prev_L <= counter:
                out = self.relu(out + shortcut)
            counter += 1
            if counter == L:
                return out, counter, True
        return out, counter, False


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion,affine=True,track_running_stats=True),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion,affine=True,track_running_stats=True)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
    
        return self.relu(self.residual_function(x) + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self, block, num_block, num_classes=100):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64,affine=True,track_running_stats=True),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Linear(512 * block.expansion, num_classes, bias=True)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """
        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, thresholds=0, L=0, t=0, prev_L=0):
        out = x
        counter = 0
        for i in range(len(self.conv1)):
            if 'SPIKE_layer' in str(self.conv1[i]):
                if prev_L <= counter:
                    out = self.conv1[i](out, t)
            else:
                if prev_L <= counter:
                    out = self.conv1[i](out)
            if 'SPIKE_layer' in str(self.conv1[i]):
                counter += 1
                if counter == L:
                    return out
            if 'ReLU' in str(self.conv1[i]):
                counter += 1
                if counter == L:
                    return out
        for i in range(len(self.conv2_x)):
            out, counter, found = self.conv2_x[i](out, counter, L, t, prev_L)
            if found:
                return out
        for i in range(len(self.conv3_x)):
            out, counter, found = self.conv3_x[i](out, counter, L, t, prev_L)
            if found:
                return out
        for i in range(len(self.conv4_x)):
            out, counter, found = self.conv4_x[i](out, counter, L, t, prev_L)
            if found:
                return out
        for i in range(len(self.conv5_x)):
            out, counter, found = self.conv5_x[i](out, counter, L, t, prev_L)
            if found:
                return out
        if prev_L <= counter:
            out = self.avg_pool(out)

            if len(out.shape) == 4:
                out = out.view(out.size(0), -1)
            elif len(out.shape) == 5:
                out = out.view(out.size(0), out.size(1), out.size(2))
            else:
                raise NotImplementedError

            out = self.fc(out)
        return out


class ResNet4Cifar(nn.Module):
    def __init__(self, block, num_block, num_classes=10, add_linear=False,
                 width_mult=1):
        super().__init__()
        self.width_mult = width_mult
        self.in_channels = 16 * width_mult
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16*width_mult, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(16*width_mult),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16*width_mult, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 32*width_mult, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 64*width_mult, num_block[2], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.add_linear = add_linear
        if not add_linear:
            self.fc = nn.Linear(64 * block.expansion * width_mult, num_classes)
        else:
            fc = [nn.Linear(64 * block.expansion * width_mult, 256),
                  nn.ReLU(inplace=True),
                  nn.Linear(256, num_classes)]
            fc = nn.Sequential(*fc)
            self.fc = fc

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, thresholds=0, L=0, t=0, prev_L=0):
        out = x
        counter = 0
        for i in range(len(self.conv1)):
            if 'SPIKE_layer' in str(self.conv1[i]):
                if prev_L <= counter:
                    out = self.conv1[i](out, t)
            else:
                if prev_L <= counter:
                    out = self.conv1[i](out)
            if 'SPIKE_layer' in str(self.conv1[i]):
                counter += 1
                if counter == L:
                    return out
            if 'ReLU' in str(self.conv1[i]):
                counter += 1
                if counter == L:
                    return out

        for i in range(len(self.conv2_x)):
            out, counter, found = self.conv2_x[i](out, counter, L, t, prev_L)
            if found:
                return out
        for i in range(len(self.conv3_x)):
            out, counter, found = self.conv3_x[i](out, counter, L, t, prev_L)
            if found:
                return out
        for i in range(len(self.conv4_x)):
            out, counter, found = self.conv4_x[i](out, counter, L, t, prev_L)
            if found:
                return out

        if prev_L <= counter:
            out = self.avg_pool(out)
            if not self.add_linear:
                out = out.view(out.size(0), -1)
                out = self.fc(out)
            else:
                if len(out.shape) == 4:
                    out = out.view(out.size(0), -1)
                elif len(out.shape) == 5:
                    out = out.view(out.size(0), out.size(1), out.size(2))
                else:
                    raise NotImplementedError

        for i in range(len(self.fc)):
            if 'SPIKE_layer' in str(self.fc[i]):
                if prev_L <= counter:
                    out = self.fc[i](out, t)
                counter += 1
                if counter == L:
                    return out
            else:
                if prev_L <= counter:
                    out = self.fc[i](out)

            if 'LIFSpike' in str(self.fc[i]):
                counter += 1
                if counter == L:
                    return out
            if 'ReLU' in str(self.fc[i]):
                counter += 1
                if counter == L:
                    return out
        return out


def resnet18(num_classes=10, **kargs):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def resnet19(num_classes=10, **kargs):
    """ return a ResNet 18 object
    """
    return ResNet4Cifar(BasicBlock, [3, 3, 2], num_classes=num_classes,
                        add_linear=True, width_mult=8)


def resnet20(num_classes=10, **kargs):
    """ return a ResNet 20 object
    """
    return ResNet4Cifar(BasicBlock, [3, 3, 3], num_classes=num_classes)


def resnet34(num_classes=10, **kargs):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def resnet50(num_classes=10, **kargs):
    """ return a ResNet 50 object
    """
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


def resnet101(num_classes=10, **kargs):
    """ return a ResNet 101 object
    """
    return ResNet(BottleNeck, [3, 4, 23, 3],num_classes=num_classes)


def resnet152(num_classes=10, **kargs):
    """ return a ResNet 152 object
    """
    return ResNet(BottleNeck, [3, 8, 36, 3],num_classes=num_classes)
