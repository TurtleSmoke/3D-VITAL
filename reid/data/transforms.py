import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


class ToTensorNoScale:
    """PIL image -> float tensor kept in [0, 255]. The model normalizes to [-1, 1] itself."""

    def __call__(self, pic):
        return torch.from_numpy(np.array(pic, copy=True)).permute(2, 0, 1).contiguous().float()

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SynchronizedTransform:
    """The train augmentations below, with the same resize, crop and flip applied to the visibility map (Sec. 3.2)."""

    def __init__(self, height, width):
        self.size = [height, width]
        self.erasing = T.RandomErasing(p=0.5, value=[123.675, 116.28, 103.53])  # ImageNet mean, in [0, 255]

    def __call__(self, image, visibility):
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        visibility = TF.resize(visibility, self.size, interpolation=InterpolationMode.NEAREST)
        image, visibility = TF.pad(image, 10), TF.pad(visibility, 10)
        i, j, h, w = T.RandomCrop.get_params(image, self.size)
        image, visibility = TF.crop(image, i, j, h, w), TF.crop(visibility, i, j, h, w)
        if torch.rand(1) < 0.5:
            image, visibility = TF.hflip(image), TF.hflip(visibility)
        return self.erasing(ToTensorNoScale()(image)), torch.from_numpy(np.array(visibility) > 127).float()


def build_transforms(img_size, is_train=True):
    """Standard TransReID / BoT augmentations."""
    height, width = img_size
    if not is_train:
        return T.Compose([T.Resize([height, width]), ToTensorNoScale()])

    return SynchronizedTransform(height, width)
