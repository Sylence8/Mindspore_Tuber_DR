import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal

class hswish(nn.Cell):
    def __init__(self):
        super(hswish, self).__init__()
        self.relu = ops.ReLU6()

    def construct(self, x):
        out = x * self.relu(x + 3) / 6
        return out

class hsigmoid(nn.Cell):
    def __init__(self):
        super(hsigmoid, self).__init__()
        self.relu = ops.ReLU6()

    def construct(self, x):
        out = self.relu(x + 3) / 6
        return out

class SeModule(nn.Cell):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.SequentialCell([
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, pad_mode='same', has_bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(),
            nn.Conv2d(expand_size, in_size, kernel_size=1, pad_mode='same', has_bias=False),
            hsigmoid()
        ])

    def construct(self, x):
        return x * self.se(x)

class Block(nn.Cell):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, has_bias=False, pad_mode='same')
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act()

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, group=expand_size, has_bias=False, pad_mode='same')
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act()
        self.se = SeModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, has_bias=False, pad_mode='same')
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act()

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.SequentialCell([
                nn.Conv2d(in_size, out_size, kernel_size=1, has_bias=False, pad_mode='same'),
                nn.BatchNorm2d(out_size)
            ])

        if stride == 2 and in_size != out_size:
            self.skip = nn.SequentialCell([
                nn.Conv2d(in_size, in_size, kernel_size=3, stride=stride, group=in_size, has_bias=False, pad_mode='same'),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, has_bias=False, pad_mode='same'),
                nn.BatchNorm2d(out_size)
            ])

        if stride == 2 and in_size == out_size:
            self.skip = nn.SequentialCell([
                nn.Conv2d(in_size, out_size, kernel_size=3, stride=stride, group=in_size, has_bias=False, pad_mode='same'),
                nn.BatchNorm2d(out_size)
            ])

    def construct(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)

class MobileNetV3_Small(nn.Cell):
    def __init__(self, num_classes=1000, act=hswish):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, pad_mode='same', has_bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act()

        self.bneck = nn.SequentialCell([
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        ])

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, pad_mode='same', has_bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Dense(576, 1280, has_bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act()
        self.drop = nn.Dropout(0.2)
        self.linear4 = nn.Dense(1280, num_classes, weight_init=HeNormal(1280))

    def construct(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten()
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        return self.linear4(out)

class MobileNetV3_Large(nn.Cell):
    def __init__(self, num_classes=1000, act=hswish):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, pad_mode='same', has_bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act()

        self.bneck = nn.SequentialCell([
            Block(3, 16, 16, 16, nn.ReLU, False, 1),
            Block(3, 16, 64, 24, nn.ReLU, False, 2),
            Block(3, 24, 72, 24, nn.ReLU, False, 1),
            Block(5, 24, 72, 40, nn.ReLU, True, 2),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(5, 40, 120, 40, nn.ReLU, True, 1),
            Block(3, 40, 240, 80, act, False, 2),
            Block(3, 80, 200, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 184, 80, act, False, 1),
            Block(3, 80, 480, 112, act, True, 1),
            Block(3, 112, 672, 112, act, True, 1),
            Block(5, 112, 672, 160, act, True, 2),
            Block(5, 160, 672, 160, act, True, 1),
            Block(5, 160, 960, 160, act, True, 1),
        ])

        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, pad_mode='same', has_bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = act()
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Dense(960, 1280, has_bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act()
        self.drop = nn.Dropout(0.2)

        self.linear4 = nn.Dense(1280, num_classes, weight_init=HeNormal(1280))

    def construct(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)

        out = self.hs2(self.bn2(self.conv2(out)))
        out = self.gap(out).flatten()
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))

        return self.linear4(out)
