import numbers
import random
import numpy as np
from PIL import Image, ImageOps

import torch
# ===============================img gt together tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, density, dot=None):
        if dot is None:
            for t in self.transforms:
                img, density = t(img, density)
            return img, density
        for t in self.transforms:
            img, density, dot = t(img, density, dot)
        return img, density, dot

class RandomHorizontallyFlip(object):
    def __call__(self, img, density, dot=None):
        if random.random() < 0.5:
            if dot is None:
                return img.transpose(Image.FLIP_LEFT_RIGHT), density.transpose(Image.FLIP_LEFT_RIGHT)
            return img.transpose(Image.FLIP_LEFT_RIGHT), density.transpose(Image.FLIP_LEFT_RIGHT), dot.transpose(Image.FLIP_LEFT_RIGHT),
        if dot is None:
            return img, density
        return img, density, dot

class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, density, dot=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            density = ImageOps.expand(density, border=self.padding, fill=0)
            if dot!= None:
                dot = ImageOps.expand(dot, border=self.padding, fill=0)
        assert img.size == density.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            if dot== None:
                return img, density
            else:
                return img, density, dot
        if w < tw or h < th:
            if dot==None:
                return img.resize((tw, th), Image.BILINEAR), \
                       density.resize((tw, th), Image.NEAREST)
            else:
                return img.resize((tw, th), Image.BILINEAR), \
                       density.resize((tw, th), Image.NEAREST), \
                       dot.resize((tw, th), Image.NEAREST)
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        if dot==None:
            return img.crop((x1, y1, x1 + tw, y1 + th)), \
                   density.crop((x1, y1, x1 + tw, y1 + th))
        else:
            return img.crop((x1, y1, x1 + tw, y1 + th)), \
                   density.crop((x1, y1, x1 + tw, y1 + th)), \
                   dot.crop((x1, y1, x1 + tw, y1 + th))


class FreeScale(object):
    def __init__(self, size):
        self.size = size  # (h, w)

    def __call__(self, img, density, dot=None):
        if dot==None:
            return img.resize((self.size[1], self.size[0]), Image.BILINEAR), \
                   density.resize((self.size[1], self.size[0]), Image.NEAREST)
        else:
            return img.resize((self.size[1], self.size[0]), Image.BILINEAR), \
                   density.resize((self.size[1], self.size[0]), Image.NEAREST), \
                   dot.resize((self.size[1], self.size[0]), Image.NEAREST)


# ===============================density,dot map tranforms============================

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor

class LabelNormalize(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, tensor):
        tensor = torch.from_numpy(np.array(tensor))
        tensor = tensor*self.para
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, density,dot=None):
        w, h = density.size
        if self.factor==1:
            return density
        density = density.resize((w//self.factor, h//self.factor), Image.BICUBIC)*self.factor*self.factor

        return density
