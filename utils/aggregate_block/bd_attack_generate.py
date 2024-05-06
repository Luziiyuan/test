# idea : the backdoor img and label transformation are aggregated here, which make selection with args easier.

import sys, logging
sys.path.append('../../')
import imageio
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

from utils.bd_img_transform.lc import labelConsistentAttack
from utils.bd_img_transform.blended import blendedImageAttack
from utils.bd_img_transform.patch import AddMaskPatchTrigger, SimpleAdditiveTrigger
from utils.bd_img_transform.sig import sigTriggerAttack
from utils.bd_img_transform.SSBA import SSBA_attack_replace_version
from utils.bd_label_transform.backdoor_label_transform import *
from torchvision.transforms import Resize

class general_compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img, *args, **kwargs):
        for transform, if_all in self.transform_list:
            if if_all == False:
                img = transform(img)
            else:
                img = transform(img, *args, **kwargs)
        return img

class convertNumpyArrayToFloat32(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np_img_float32.astype(np.float32)
npToFloat32 = convertNumpyArrayToFloat32()

class clipAndConvertNumpyArrayToUint8(object):
    def __init__(self):
        pass
    def __call__(self, np_img_float32):
        return np.clip(np_img_float32, 0, 255).astype(np.uint8)
npClipAndToUint8 = clipAndConvertNumpyArrayToUint8()

def bd_attack_img_trans_generate(args):
    '''
    # idea : use args to choose which backdoor img transform you want
    :param args: args that contains parameters of backdoor attack
    :return: transform on img for backdoor attack in both train and test phase
    '''

    if args.attack in ['badnet',]:
        trans = transforms.Compose([
            transforms.Resize(args.img_size[:2]),  # (32, 32)
            np.array,
        ])

        bd_transform = AddMaskPatchTrigger(
            trans(Image.open(args.patch_mask_path)),
        )

        train_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
        ])

        test_bd_transform = general_compose([
            (transforms.Resize(args.img_size[:2]), False),
            (np.array, False),
            (bd_transform, True),
            (npClipAndToUint8,False),
        ])



    return train_bd_transform, test_bd_transform


def bd_attack_label_trans_generate(args):
    '''
    # idea : use args to choose which backdoor label transform you want
    from args generate backdoor label transformation

    '''
    if args.attack_label_trans == 'all2one':
        target_label = int(args.attack_target)
        bd_label_transform = AllToOne_attack(target_label)
    elif args.attack_label_trans == 'all2all':
        bd_label_transform = AllToAll_shiftLabelAttack(
            int(1 if "attack_label_shift_amount" not in args.__dict__ else args.attack_label_shift_amount),
            int(args.num_classes)
        )

    return bd_label_transform
