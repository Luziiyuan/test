# idea: select model you use in training and the trainer (the warper for training process)

import logging
import sys

sys.path.append('../../')

import torch
import torchvision.models as models
from torchvision.models.resnet import resnet34, resnet50
from typing import Optional
from torchvision.transforms import Resize

from utils.trainer_cls import ModelTrainerCLS


def partially_load_state_dict(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        try:
            param = param.data
            own_state[name].copy_(param)
        except:
            print(f"unmatch: {name}")
            continue


# trainer is cls
def generate_cls_model(
        model_name: str,
        num_classes: int = 10,
        image_size: int = 32,
        **kwargs,
):
    '''
    # idea: aggregation block for selection of classifcation models
    :param model_name:
    :param num_classes:
    :return:
    '''

    if model_name == 'resnet18':
        from torchvision.models.resnet import resnet18
        net = resnet18(num_classes=num_classes, **kwargs)
    elif model_name == 'preactresnet18':
        logging.debug('Make sure you want PreActResNet18, which is NOT resnet18.')
        from models.preact_resnet import PreActResNet18
        if kwargs.get("pretrained", False):
            logging.warning("PreActResNet18 pretrained on cifar10, NOT ImageNet!")
            net_from_cifar10 = PreActResNet18()  # num_classes = num_classes)
            net_from_cifar10.load_state_dict(
                torch.load("../resource/trojannn/clean_preactresnet18.pt", map_location="cpu"
                           )['model_state_dict']
            )
            net = PreActResNet18(num_classes=num_classes)
            partially_load_state_dict(net, net_from_cifar10.state_dict())
        else:
            net = PreActResNet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        net = resnet34(num_classes=num_classes, **kwargs)
    elif model_name == 'resnet50':
        net = resnet50(num_classes=num_classes, **kwargs)
    elif model_name == "vgg11":
        net = models.vgg11(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg16':
        net = models.vgg16(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg19':
        net = models.vgg19(num_classes=num_classes, **kwargs)
    elif model_name == 'vgg19_bn':
        if kwargs.get("pretrained", False):
            net_from_imagenet = models.vgg19_bn(pretrained=True)  # num_classes = num_classes)
            net = models.vgg19_bn(num_classes=num_classes, **{k: v for k, v in kwargs.items() if k != "pretrained"})
            partially_load_state_dict(net, net_from_imagenet.state_dict())
        else:
            net = models.vgg19_bn(num_classes=num_classes, **kwargs)
    else:
        raise SystemError('NO valid model match in function generate_cls_model!')

    return net


def generate_cls_trainer(
        model,
        attack_name: Optional[str] = None,
        amp: bool = False,
):
    '''
    # idea: The warpper of model, which use to receive training settings.
        You can add more options for more complicated backdoor attacks.

    :param model:
    :param attack_name:
    :return:
    '''

    trainer = ModelTrainerCLS(
        model=model,
        amp=amp,
    )

    return trainer
