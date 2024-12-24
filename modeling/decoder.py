import torch
import torch.nn as nn
import torch.nn.functional as F
from DCREN.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from CDCREN.modeling.HT import hough_transform, CAT_HTIHT

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters, BatchNorm, inp=False):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.bn1 = BatchNorm(in_channels // 4)
        self.relu1 = nn.ReLU()
        self.inp = inp

        self.deconv1 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )
        self.deconv2 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv3 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (9, 1), padding=(4, 0)
        )
        self.deconv4 = nn.Conv2d(
            in_channels // 4, in_channels // 8, (1, 9), padding=(0, 4)
        )

        self.bn2 = BatchNorm(in_channels // 4 + in_channels // 4)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(
            in_channels // 4 + in_channels // 4, n_filters, 1)
        self.bn3 = BatchNorm(n_filters)
        self.relu3 = nn.ReLU()

        self._init_weight()

    def forward(self, x, inp = False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.deconv1(x)
        x2 = self.deconv2(x)
        x3 = self.inv_h_transform(self.deconv3(self.h_transform(x)))
        x4 = self.inv_v_transform(self.deconv4(self.v_transform(x)))
        x = torch.cat((x1, x2, x3, x4), 1)
        if self.inp:
            x = F.interpolate(x, scale_factor=2)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.ConvTranspose2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1]))
        x = x.reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        x = x.reshape(shape[0], shape[1], shape[2], 2*shape[3]-1)
        return x.permute(0, 1, 3, 2)

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2)
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1)
        x = torch.nn.functional.pad(x, (0, shape[-2]))
        x = x.reshape(shape[0], shape[1], shape[-2], 2*shape[-2])
        x = x[..., 0: shape[-2]]
        return x.permute(0, 1, 3, 2)

class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            in_inplanes = 256
        else:
            raise NotImplementedError

        filters = [256, 512, 1024, 2048]
        ht_channels = 16

        vote_index_1 = hough_transform(rows=128, cols=128, theta_res=3, rho_res=1)
        vote_index_2 = hough_transform(rows=64, cols=64, theta_res=3, rho_res=1)
        vote_index_3 = hough_transform(rows=64, cols=64, theta_res=3, rho_res=1)
        vote_index_1 = torch.from_numpy(vote_index_1).float().contiguous().cuda()
        vote_index_2 = torch.from_numpy(vote_index_2).float().contiguous().cuda()
        vote_index_3 = torch.from_numpy(vote_index_3).float().contiguous().cuda()

        self.ht_1 = CAT_HTIHT(vote_index_1, inplanes=filters[0], outplanes=ht_channels)
        self.ht_2 = CAT_HTIHT(vote_index_2, inplanes=filters[1], outplanes=ht_channels)
        self.ht_3 = CAT_HTIHT(vote_index_3, inplanes=filters[2], outplanes=ht_channels)

        self.conv = nn.Conv2d(in_channels=256, out_channels=2048, kernel_size=1)
        self.conv1 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1)

        self.decoder4 = DecoderBlock(in_inplanes, 256, BatchNorm)
        self.decoder3 = DecoderBlock(512, 128, BatchNorm)
        self.decoder2 = DecoderBlock(256, 64, BatchNorm, inp=True)
        self.decoder1 = DecoderBlock(128, 64, BatchNorm, inp=True)

        self.conv_e3 = nn.Sequential(nn.Conv2d(1024, 256, 1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU())

        self.conv_e2 = nn.Sequential(nn.Conv2d(512, 128, 1, bias=False),
                                     BatchNorm(128),
                                     nn.ReLU())

        self.conv_e1 = nn.Sequential(nn.Conv2d(256, 64, 1, bias=False),
                                     BatchNorm(64),
                                     nn.ReLU())

        self._init_weight()


    def forward(self, e1, e2, e3, e4):
        d4 = torch.cat((self.decoder4(e4), self.conv_e3(e3)), dim=1) + 0.5 * self.conv2(self.ht_3(e3))
        d3 = torch.cat((self.decoder3(d4), self.conv_e2(e2)), dim=1) + 0.5 * self.conv4(self.ht_2(e2))
        d2 = torch.cat((self.decoder2(d3), self.conv_e1(e1)), dim=1) + 0.5 * self.conv5(self.ht_1(e1))
        d1 = self.decoder1(d2)
        x = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=True)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)