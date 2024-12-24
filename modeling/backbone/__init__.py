from DCREN.modeling.backbone import resnet

def build_backbone(backbone, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet50(output_stride, BatchNorm)
    else:
        raise NotImplementedError
