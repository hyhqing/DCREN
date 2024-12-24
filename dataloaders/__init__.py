from DCREN.dataloaders.datasets import zurich, zurich_crop, mass, mass_crop
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def make_data_loader(args, **kwargs):

    if args.dataset == 'zurich':
        train_set = zurich_crop.Segmentation(args, split='train')
        val_set = zurich_crop.Segmentation(args, split='val')
        test_set = zurich.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'mass':
        train_set = mass_crop.Segmentation(args, split='train')
        val_set = mass_crop.Segmentation(args, split='val')
        test_set = mass.Segmentation(args, split='test')

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoaderX(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoaderX(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoaderX(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError

