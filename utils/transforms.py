import torchvision.transforms.functional as TF
from gutils.basic import torch_randint


def d4_rotation(img, id):
    rot = id % 4
    flip = id // 4
    img = TF.rotate(img, 90 * rot)
    if flip:
        img = TF.hflip(img)
    return img


class RandomD4Rotation:
    def __call__(self, img):
        return d4_rotation(img, torch_randint(0, 8))
