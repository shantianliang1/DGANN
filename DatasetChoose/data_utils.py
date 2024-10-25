def dataset_builder(dataset):
    if dataset == 'mini':
        from DatasetChoose.mini_Imagenet import MiniImagenetLoader as Dataset
    elif dataset == 'cifar':
        from DatasetChoose.cifar_fs import CifarFsLoader as Dataset

    elif dataset == 'tiered':
        from DatasetChoose.tiered_Imagenet import TieredImagenetLoader as Dataset
    else:
        raise ValueError('Unknown Dataset')
    return Dataset
