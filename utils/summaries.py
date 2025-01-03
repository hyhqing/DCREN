import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, target1, output, global_step, split):
        grid_image = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
        writer.add_image(split + 'Image', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image(split + 'Predicted label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image(split + 'Groundtruth label', grid_image, global_step)
        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target1[:3], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image(split + 'Groundtruth1 label', grid_image, global_step)