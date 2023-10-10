import mindspore.nn as nn
from mindspore.common.initializer import HeNormal
from mindspore.ops import operations as P

class _DenseLayer(nn.Cell):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, has_bias=False, pad_mode="pad", padding=0)
        self.bn1 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, pad_mode="pad", padding=1, has_bias=False)
        self.bn2 = nn.BatchNorm3d(growth_rate)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(keep_prob=1 - drop_rate)

    def construct(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        return out

class _DenseBlock(nn.Cell):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            new_layer = _DenseLayer(num_input_features, growth_rate, bn_size, drop_rate)
            layers.append(new_layer)
            num_input_features += growth_rate
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        features = self.layers(x)
        return features

class _Transition(nn.Cell):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.conv = nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, has_bias=False, pad_mode="valid")
        self.bn = nn.BatchNorm3d(num_output_features)
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), pad_mode="same")

    def construct(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        return out

class DenseNet(nn.Cell):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(DenseNet, self).__init__()
        self.num_classes = num_classes

        self.features = nn.SequentialCell([
            nn.Conv3d(in_channels=3, out_channels=num_init_features, kernel_size=7, stride=2, pad_mode="same", padding=3, has_bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=3, stride=2, pad_mode="same"),
        ])

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add(block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
                self.features.add(trans)
                num_features = num_features // 2

        self.features.add(nn.BatchNorm3d(num_features))
        self.features.add(nn.ReLU())
        self.features.add(nn.AvgPool3d(kernel_size=7, stride=1))

        self.classifier = nn.Dense(num_features, num_classes, weight_init=HeNormal(), has_bias=True)

    def construct(self, x):
        x = self.features(x)
        x = P.Reshape()(x, (P.Shape()(x)[0], -1))
        x = self.classifier(x)
        return x

def densenet121(**kwargs):
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)

def densenet169(**kwargs):
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)

def densenet201(**kwargs):
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)

def densenet264(**kwargs):
    return DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 64, 48), **kwargs)
