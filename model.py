import os
from mindspore import nn
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common.initializer import initializer
from models import resnet, mobilenetv3, resnext, densenet
from models.resnet import get_fine_tuning_parameters

context.set_context(mode=context.GRAPH_MODE, device_target="GPU")  # 设置运行模式和设备类型


def generate_model(opt):
    assert opt.model in ['resnet', 'densenet', 'mobilenetv3_s', 'mobilenetv3_l', 'resnext']

    if opt.model == 'mobilenetv3_s':
        model = mobilenetv3.MobileNetV3_Small(num_classes=opt.n_classes, act=mobilenetv3.hswish)

    elif opt.model == 'mobilenetv3_l':
        model = mobilenetv3.MobileNetV3_Large(num_classes=opt.n_classes, act=mobilenetv3.hswish)

    elif opt.model == 'resnet':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if opt.model_depth == 10:
            model = resnet.resnet10(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 18:
            model = resnet.resnet18(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 34:
            model = resnet.resnet34(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 50:
            model = resnet.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 101:
            model = resnet.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 152:
            model = resnet.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 200:
            model = resnet.resnet200(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)

    elif opt.model == 'resnext':
        assert opt.model_depth in [50, 101, 152]

        if opt.model_depth == 50:
            model = resnext.resnet50(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 101:
            model = resnext.resnet101(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 152:
            model = resnext.resnet152(
                num_classes=opt.n_classes,
                shortcut_type=opt.resnet_shortcut,
                cardinality=opt.resnext_cardinality,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)

    elif opt.model == 'densenet':
        assert opt.model_depth in [121, 169, 201, 264]

        if opt.model_depth == 121:
            model = densenet.densenet121(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 169:
            model = densenet.densenet169(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 201:
            model = densenet.densenet201(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)
        elif opt.model_depth == 264:
            model = densenet.densenet264(
                num_classes=opt.n_classes,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration,
                channels=opt.channels)

    if not opt.no_cuda:
        model = model.to("cuda")
        model = nn.DataParallel(model, device_ids=None)

        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            param_dict = load_checkpoint(opt.pretrain_path)
            model.load_parameters(param_dict)

            if opt.model == 'densenet':
                model.module.classifier = nn.Dense(
                    model.module.classifier.in_channels, opt.n_finetune_classes)
                model.module.classifier.to("cuda")
            else:
                model.module.fc = nn.Dense(model.module.fc.in_channels, opt.n_finetune_classes)
                model.module.fc.to("cuda")

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters
    else:
        if opt.pretrain_path:
            print('loading pretrained model {}'.format(opt.pretrain_path))
            param_dict = load_checkpoint(opt.pretrain_path)
            model.load_parameters(param_dict)

            if opt.model == 'densenet':
                model.classifier = nn.Dense(
                    model.classifier.in_channels, opt.n_finetune_classes)
            else:
                model.fc = nn.Dense(model.fc.in_channels, opt.n_finetune_classes)

            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
            return model, parameters

    return model, model.parameters()


def generate_cammodel(config):
    if config.model_depth == 10:
        model = resnet.resnet10(
            num_classes=config.n_classes,
            shortcut_type=config.resnet_shortcut,
            sample_size=config.sample_size,
            sample_duration=config.sample_duration,
            channels=config.channels)
    elif config.model_depth == 18:
        model = resnet.resnet18(
            num_classes=config.n_classes,
            shortcut_type=config.resnet_shortcut,
            sample_size=config.sample_size,
            sample_duration=config.sample_duration,
            channels=config.channels)
    elif config.model_depth == 34:
        model = resnet.resnet34(
            num_classes=config.n_classes,
            shortcut_type=config.resnet_shortcut,
            sample_size=config.sample_size,
            sample_duration=config.sample_duration,
            channels=config.channels)
    elif config.model_depth == 50:
        model = resnet.resnet50(
            num_classes=config.n_classes,
            shortcut_type=config.resnet_shortcut,
            sample_size=config.sample_size,
            sample_duration=config.sample_duration,
            channels=config.channels)

    if not config.no_cuda:
        model = model.to("cuda")
    return model, model.parameters()
