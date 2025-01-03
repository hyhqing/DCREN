import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        con0 = sample['connect0']
        con1 = sample['connect1']
        con2 = sample['connect2']
        con_d1_0 = sample['connect_d1_0']
        con_d1_1 = sample['connect_d1_1']
        con_d1_2 = sample['connect_d1_2']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        mask1 = np.array(mask1).astype(np.float32)
        con0 = np.array(con0).astype(np.float32)
        con1 = np.array(con1).astype(np.float32)
        con2 = np.array(con2).astype(np.float32)

        con_d1_0 = np.array(con_d1_0).astype(np.float32)
        con_d1_1 = np.array(con_d1_1).astype(np.float32)
        con_d1_2 = np.array(con_d1_2).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std
        mask /= 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        mask1 /= 255.0
        mask1[mask1 >= 0.5] = 1
        mask1[mask1 <= 0.5] = 0
        con0 /= 255.0
        con1 /= 255.0
        con2 /= 255.0

        con_d1_0 /= 255.0
        con_d1_1 /= 255.0
        con_d1_2 /= 255.0


        return {'image': img,
                'label': mask, 'label1': mask1, 'connect0': con0, 'connect1': con1, 'connect2': con2,
                'connect_d1_0': con_d1_0, 'connect_d1_1': con_d1_1, 'connect_d1_2': con_d1_2}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        con0 = sample['connect0']
        con1 = sample['connect1']
        con2 = sample['connect2']

        con_d1_0 = sample['connect_d1_0']
        con_d1_1 = sample['connect_d1_1']
        con_d1_2 = sample['connect_d1_2']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        con0 = np.array(con0).astype(np.float32).transpose((2, 0, 1))
        con1 = np.array(con1).astype(np.float32).transpose((2, 0, 1))
        con2 = np.array(con2).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        mask1 = np.array(mask1).astype(np.float32)
        con0 = np.array(con0).astype(np.float32)
        con1 = np.array(con1).astype(np.float32)
        con2 = np.array(con2).astype(np.float32)

        con_d1_0 = np.array(con_d1_0).astype(np.float32).transpose((2, 0, 1))
        con_d1_1 = np.array(con_d1_1).astype(np.float32).transpose((2, 0, 1))
        con_d1_2 = np.array(con_d1_2).astype(np.float32).transpose((2, 0, 1))
        con_d1_0 = np.array(con_d1_0).astype(np.float32)
        con_d1_1 = np.array(con_d1_1).astype(np.float32)
        con_d1_2 = np.array(con_d1_2).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        mask1 = torch.from_numpy(mask1).float()
        con0 = torch.from_numpy(con0).float()
        con1 = torch.from_numpy(con1).float()
        con2 = torch.from_numpy(con2).float()

        con_d1_0 = torch.from_numpy(con_d1_0).float()
        con_d1_1 = torch.from_numpy(con_d1_1).float()
        con_d1_2 = torch.from_numpy(con_d1_2).float()

        return {'image': img,
                'label': mask, 'label1': mask1, 'connect0': con0, 'connect1': con1, 'connect2': con2,
                'connect_d1_0': con_d1_0, 'connect_d1_1': con_d1_1, 'connect_d1_2': con_d1_2}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        con0 = sample['connect0']
        con1 = sample['connect1']
        con2 = sample['connect2']

        con_d1_0 = sample['connect_d1_0']
        con_d1_1 = sample['connect_d1_1']
        con_d1_2 = sample['connect_d1_2']

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
            con0 = con0.transpose(Image.FLIP_LEFT_RIGHT)
            con1 = con1.transpose(Image.FLIP_LEFT_RIGHT)
            con2 = con2.transpose(Image.FLIP_LEFT_RIGHT)

            con_d1_0 = con_d1_0.transpose(Image.FLIP_LEFT_RIGHT)
            con_d1_1 = con_d1_1.transpose(Image.FLIP_LEFT_RIGHT)
            con_d1_2 = con_d1_2.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask, 'label1': mask1, 'connect0': con0, 'connect1': con1, 'connect2': con2,
                'connect_d1_0': con_d1_0, 'connect_d1_1': con_d1_1, 'connect_d1_2': con_d1_2}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        con0 = sample['connect0']
        con1 = sample['connect1']
        con2 = sample['connect2']

        con_d1_0 = sample['connect_d1_0']
        con_d1_1 = sample['connect_d1_1']
        con_d1_2 = sample['connect_d1_2']

        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)
        mask1 = mask1.rotate(rotate_degree, Image.NEAREST)
        con0 = con0.rotate(rotate_degree, Image.NEAREST)
        con1 = con1.rotate(rotate_degree, Image.NEAREST)
        con2 = con2.rotate(rotate_degree, Image.NEAREST)

        con_d1_0 = con_d1_0.rotate(rotate_degree, Image.NEAREST)
        con_d1_1 = con_d1_1.rotate(rotate_degree, Image.NEAREST)
        con_d1_2 = con_d1_2.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask, 'label1': mask1, 'connect0': con0, 'connect1': con1, 'connect2': con2,
                'connect_d1_0': con_d1_0, 'connect_d1_1': con_d1_1, 'connect_d1_2': con_d1_2}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        con0 = sample['connect0']
        con1 = sample['connect1']
        con2 = sample['connect2']

        con_d1_0 = sample['connect_d1_0']
        con_d1_1 = sample['connect_d1_1']
        con_d1_2 = sample['connect_d1_2']

        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask, 'label1': mask1, 'connect0': con0, 'connect1': con1, 'connect2': con2,
                'connect_d1_0': con_d1_0, 'connect_d1_1': con_d1_1, 'connect_d1_2': con_d1_2}


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        con0 = sample['connect0']
        con1 = sample['connect1']
        con2 = sample['connect2']

        con_d1_0 = sample['connect_d1_0']
        con_d1_1 = sample['connect_d1_1']
        con_d1_2 = sample['connect_d1_2']

        # random scale (short edge)
        short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        mask1 = mask1.resize((ow, oh), Image.NEAREST)
        con0 = con0.resize((ow, oh), Image.NEAREST)
        con1 = con1.resize((ow, oh), Image.NEAREST)
        con2 = con2.resize((ow, oh), Image.NEAREST)

        con_d1_0 = con_d1_0.resize((ow, oh), Image.NEAREST)
        con_d1_1 = con_d1_1.resize((ow, oh), Image.NEAREST)
        con_d1_2 = con_d1_2.resize((ow, oh), Image.NEAREST)

        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
            mask1 = ImageOps.expand(mask1, border=(0, 0, padw, padh), fill=self.fill)
            con0 = ImageOps.expand(con0, border=(0, 0, padw, padh), fill=self.fill)
            con1 = ImageOps.expand(con1, border=(0, 0, padw, padh), fill=self.fill)
            con2 = ImageOps.expand(con2, border=(0, 0, padw, padh), fill=self.fill)

            con_d1_0 = ImageOps.expand(con_d1_0, border=(0, 0, padw, padh), fill=self.fill)
            con_d1_1 = ImageOps.expand(con_d1_1, border=(0, 0, padw, padh), fill=self.fill)
            con_d1_2 = ImageOps.expand(con_d1_2, border=(0, 0, padw, padh), fill=self.fill)

        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask1 = mask1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con0 = con0.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con1 = con1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con2 = con2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        con_d1_0 = con_d1_0.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con_d1_1 = con_d1_1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con_d1_2 = con_d1_2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask, 'label1': mask1, 'connect0': con0, 'connect1': con1, 'connect2': con2,
                'connect_d1_0': con_d1_0, 'connect_d1_1': con_d1_1, 'connect_d1_2': con_d1_2}


class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        con0 = sample['connect0']
        con1 = sample['connect1']
        con2 = sample['connect2']

        con_d1_0 = sample['connect_d1_0']
        con_d1_1 = sample['connect_d1_1']
        con_d1_2 = sample['connect_d1_2']

        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        mask1 = mask1.resize((ow, oh), Image.NEAREST)
        con0 = con0.resize((ow, oh), Image.NEAREST)
        con1 = con1.resize((ow, oh), Image.NEAREST)
        con2 = con2.resize((ow, oh), Image.NEAREST)

        con_d1_0 = con_d1_0.resize((ow, oh), Image.NEAREST)
        con_d1_1 = con_d1_1.resize((ow, oh), Image.NEAREST)
        con_d1_2 = con_d1_2.resize((ow, oh), Image.NEAREST)

        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask1 = mask1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con0 = con0.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con1 = con1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con2 = con2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        con_d1_0 = con_d1_0.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con_d1_1 = con_d1_1.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        con_d1_2 = con_d1_2.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask, 'label1': mask1, 'connect0': con0, 'connect1': con1, 'connect2': con2,
                'connect_d1_0': con_d1_0, 'connect_d1_1': con_d1_1, 'connect_d1_2': con_d1_2}

class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        con0 = sample['connect0']
        con1 = sample['connect1']
        con2 = sample['connect2']

        con_d1_0 = sample['connect_d1_0']
        con_d1_1 = sample['connect_d1_1']
        con_d1_2 = sample['connect_d1_2']

        assert img.size == mask.size == mask1.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        mask1 = mask1.resize(self.size, Image.NEAREST)
        con0 = con0.resize(self.size, Image.NEAREST)
        con1 = con1.resize(self.size, Image.NEAREST)
        con2 = con2.resize(self.size, Image.NEAREST)

        con_d1_0 = con_d1_0.resize(self.size, Image.NEAREST)
        con_d1_1 = con_d1_1.resize(self.size, Image.NEAREST)
        con_d1_2 = con_d1_2.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask, 'label1': mask1, 'connect0': con0, 'connect1': con1, 'connect2': con2,
                'connect_d1_0': con_d1_0, 'connect_d1_1': con_d1_1, 'connect_d1_2': con_d1_2}

class FixedResize_test(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']

        assert img.size == mask.size == mask1.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)
        mask1 = mask1.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask,
                'label1': mask1}

class Normalize_test(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        mask1 = np.array(mask1).astype(np.float32)

        img /= 255.0
        img -= self.mean
        img /= self.std
        mask /= 255.0
        mask[mask >= 0.5] = 1
        mask[mask <= 0.5] = 0
        mask1 /= 255.0
        mask1[mask1 >= 0.5] = 1
        mask1[mask1 <= 0.5] = 0

        return {'image': img,
                'label': mask,
                'label1': mask1}

class ToTensor_test(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        mask1 = sample['label1']

        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)
        mask1 = np.array(mask1).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()
        mask1 = torch.from_numpy(mask1).float()

        return {'image': img,
                'label': mask,
                'label1': mask1}