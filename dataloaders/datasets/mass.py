from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from DCREN.mypath import Path
from torchvision import transforms
from DCREN.dataloaders import custom_transforms as tr

class Segmentation(Dataset):
    NUM_CLASSES = 1

    def __init__(self,
                 args,
                 base_dir=Path.db_root_dir('mass'),
                 split='train',
                 ):
        """
        :param base_dir: path to dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = Path.db_root_dir(args.dataset)
        self._image_dir = os.path.join(self._base_dir, 'crops', 'images')
        self._cat_dir = os.path.join(self._base_dir, 'crops', 'gt')
        self._cat1_dir = os.path.join(self._base_dir, 'crops', 'labels')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        self.args = args

        _splits_dir = os.path.join(self._base_dir)

        self.im_ids = []
        self.images = []
        self.categories = []
        self.categories1 = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, 'test_crops' + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line)
                _cat = os.path.join(self._cat_dir, line)
                _cat1 = os.path.join(self._cat1_dir, line )
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                assert os.path.isfile(_cat1)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                self.categories1.append(_cat1)

        assert (len(self.images) == len(self.categories) == len(self.categories1))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        if self.split[0] == 'test':
            return len(self.images)
        else:
            return len(self.images) // self.args.batch_size * self.args.batch_size


    def __getitem__(self, index):
        _img, _target, _target1 = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'label': _target, 'label1': _target1}

        for split in self.split:
            if split == "train":
                return self.transform_tr(sample)
            elif split == 'val':
                return self.transform_val(sample), self.im_ids[index]
            elif split == 'test':
                return self.transform_test(sample), self.im_ids[index]


    def _make_img_gt_point_pair(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index]).convert('L')
        _target1 = Image.open(self.categories1[index]).convert('L')

        return _img, _target, _target1

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomRotate(180),
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize_test(size=self.args.crop_size),
            tr.Normalize_test(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor_test()
        ])

        return composed_transforms(sample)


if __name__ == '__main__':
    from DCREN.dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.base_size = 512
    args.crop_size = 512
    args.batch_size = 1
    args.dataset = 'mass'

    data_train = Segmentation(args, split='train')

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=0)

    for ii, sample in enumerate(dataloader):
        for jj in range(sample["image"].size()[0]):
            img = sample['image'].numpy()
            gt = sample['label'].numpy()
            label = sample['label1'].numpy()
            tmp = np.array(gt[jj]).astype(np.uint8)
            tmp1 = np.array(label[jj]).astype(np.uint8)
            segmap = decode_segmap(tmp, dataset='mass')
            segmap1 = decode_segmap(tmp1, dataset='mass')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            plt.figure()
            plt.title('display')
            plt.subplot(211)
            plt.imshow(img_tmp)
            plt.subplot(212)
            plt.imshow(segmap)
            plt.subplot(213)
            plt.imshow(segmap1)

        if ii == 1:
            break

    plt.show(block=True)


