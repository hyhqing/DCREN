class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'zurich':
            return 'C:/Users/admin/Desktop/DCREN/data/zurich/'
        elif dataset == 'mass':
            return 'C:/Users/admin/Desktop/DCREN/data/msaa/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
