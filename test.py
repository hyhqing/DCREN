import argparse
import os
import numpy as np
import time
from modeling.DCREN import *
from dataloaders import custom_transforms as tr
from PIL import Image
from torchvision import transforms
from dataloaders.utils import *
from torchvision.utils import make_grid
from dataloaders import make_data_loader
from utils.loss import SegmentationLosses
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.

    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    from PIL import Image
    grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                     normalize=normalize, range=range, scale_each=scale_each)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im = im.resize([512, 512])
    im.save(filename)

def main():
    parser = argparse.ArgumentParser(description="PyTorch DCREN Training")
    parser.add_argument('--out-path', type=str, default='C:/Users/admin/Desktop/DCREN/run',
                        help='mask image to save')
    parser.add_argument('--backbone', type=str, default='resnet',
                        help='backbone name (default: resnet)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for test ')
    parser.add_argument('--ckpt', type=str, default='C:/Users/admin/Desktop/DCREN/run/experiment_20240708_083520/checkpoint.pth.tar',
                        help='saved model')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--loss-type', type=str, default='conloss',
                        help='loss func type')
    parser.add_argument('--workers', type=int, default=16,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--dataset', type=str, default='mass',
                        choices=['zurich', 'mass'],
                        help='dataset name')
    parser.add_argument('--base-size', type=int, default=512,
                        help='base image size. szurich:512, mass:512.')
    parser.add_argument('--crop-size', type=int, default=512,
                        help='crop image size. zurich:512 mass:512.')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    kwargs = {'num_workers': args.workers, 'pin_memory': False}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    model = DCREN(config=configs.get_r50_b16_config(),
                    in_chans=3,
                    num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    model = model.cuda()
    ckpt = torch.load(args.ckpt)
    model.load_state_dict(ckpt['state_dict'])

    out_path = os.path.join(args.out_path, 'outputs/')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    model.eval()
    tbar = tqdm(test_loader, desc='\r')
    for i, sample in enumerate(tbar):
        image, target = sample[0]['image'], sample[0]['label'],
        image = image.cpu().numpy()
        image1 = image[:, :, ::-1, :]
        image2 = image[:, :, :, ::-1]
        image3 = image[:, :, ::-1, ::-1]
        image = np.concatenate((image,image1,image2,image3), axis=0)
        image = torch.from_numpy(image).float()

        img_name = sample[1][0].split('.')[0]
        if args.cuda:
            image, target = image.cuda(), target.cuda()
        with torch.no_grad():
            output, out_connect, out_connect_d1, output1 = model(image)

        out_connect_full = []
        out_connect = out_connect.data.cpu().numpy()
        out_connect_full.append(out_connect[0, ...])
        out_connect_full.append(out_connect[1, :, ::-1, :])
        out_connect_full.append(out_connect[2, :, :, ::-1])
        out_connect_full.append(out_connect[3, :, ::-1, ::-1])
        out_connect_full = np.asarray(out_connect_full).mean(axis=0)[np.newaxis, :, :, :]
        pred_connect = np.sum(out_connect_full, axis=1)

        pred_connect[pred_connect < 0.9] = 0
        pred_connect[pred_connect >= 0.9] = 1

        out_connect_d1_full = []
        out_connect_d1 = out_connect_d1.data.cpu().numpy()
        out_connect_d1_full.append(out_connect_d1[0, ...])
        out_connect_d1_full.append(out_connect_d1[1, :, ::-1, :])
        out_connect_d1_full.append(out_connect_d1[2, :, :, ::-1])
        out_connect_d1_full.append(out_connect_d1[3, :, ::-1, ::-1])
        out_connect_d1_full = np.asarray(out_connect_d1_full).mean(axis=0)[np.newaxis, :, :, :]
        pred_connect_d1 = np.sum(out_connect_d1_full, axis=1)

        pred_connect_d1[pred_connect_d1 < 2.0] = 0
        pred_connect_d1[pred_connect_d1 >= 2.0] = 1


        pred_full = []
        pred = output.data.cpu().numpy()
        target_n = target.cpu().numpy()
        pred_full.append(pred[0, ...])
        pred_full.append(pred[1, :, ::-1, :])
        pred_full.append(pred[2, :, :, ::-1])
        pred_full.append(pred[3, :, ::-1, ::-1])
        pred_full = np.asarray(pred_full).mean(axis=0)

        pred_full[pred_full > 0.1] = 1
        pred_full[pred_full < 0.1] = 0

        pred_full1 = []
        pred1 = output1.data.cpu().numpy()
        pred_full1.append(pred1[0, ...])
        pred_full1.append(pred1[1, :, ::-1, :])
        pred_full1.append(pred1[2, :, :, ::-1])
        pred_full1.append(pred1[3, :, ::-1, ::-1])
        pred_full1 = np.asarray(pred_full1).mean(axis=0)

        pred_full1[pred_full1 > 0.1] = 1
        pred_full1[pred_full1 < 0.1] = 0

        su = pred_full + pred_connect + pred_connect_d1
        su[su > 0] = 1

        # save imgs
        out_image = make_grid(image[0,:].clone().cpu().data, 3, normalize=True)
        out_GT = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=args.dataset), 3, normalize=False, range=(0, 255))
        out_pred_label_sum = make_grid(decode_seg_map_sequence(su,
                                                       dataset=args.dataset), 3, normalize=False, range=(0, 255))

        save_image(out_image, out_path + img_name + '_sat.png')
        save_image(out_GT, out_path + img_name + '_GT' + '.png')
        save_image(out_pred_label_sum, out_path + img_name + '_pred' + '.png')

if __name__ == "__main__":
   main()