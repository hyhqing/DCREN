import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from DCREN.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h

        return out

class Connect(nn.Module):
    def __init__(self, num_classes, num_neighbor, BatchNorm, reduction=3):
        super(Connect, self).__init__()
        self.seg_branch = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64, num_classes, kernel_size=1, stride=1))

        self.connect_branch = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                        nn.ReLU(),
                                        nn.Conv2d(64, num_neighbor, 3, padding=1, dilation=1),
                                            )
        self.CoordAtt = CoordAtt(num_neighbor, num_neighbor)
        self.connect_branch_d1 = nn.Sequential(nn.Conv2d(64, 64, 3, stride=1, padding=1),
                                            nn.ReLU(),
                                            nn.Conv2d(64, num_neighbor, 3, padding=3, dilation=3),
                                               )
        self.CoordAtt1 = CoordAtt(num_neighbor, num_neighbor)
        self._init_weight()

    def forward(self, input):
        seg = self.seg_branch(input)
        con = self.connect_branch(input)
        con0 = self.CoordAtt(con)
        con_d1 = self.connect_branch_d1(input)
        con1 = self.CoordAtt1(con_d1)

        return torch.sigmoid(seg), con0, con1

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

def build_connect(num_classes, num_neighbor, BatchNorm):
    return Connect(num_classes, num_neighbor, BatchNorm)