from .getdataloader import *
from .getdataset import *

def datapool(DATANAME, batchsize, num_workers, train_test_split=-1, shuffle=True):
    if DATANAME.lower() == 'cifar10':
        return GetCifar10(batchsize, num_workers, train_test_split, shuffle)
    elif DATANAME.lower() == 'cifar100':
        return GetCifar100(batchsize, num_workers,shuffle)
    elif DATANAME.lower() == 'mnist':
        return GetMnist(batchsize, num_workers)
    elif DATANAME.lower() == 'fashion':
        return GetFashion(batchsize, num_workers)
    
    else:
        print("still not support this model")
        exit(0)


def load_dataset(dataset: str, logger: logging.Logger) -> Any:
    return get_dataset(dataset, logger)

def split_dataset(dataset_size: int, num_reference_models: int):
    return split_dataset_for_training(dataset_size, num_reference_models)


def get_dataloader_from_dataset(
    dataset: torchvision.datasets,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Function to get DataLoader.

    Args:
        dataset (torchvision.datasets): The whole dataset.
        batch_size (int): Batch size for getting signals.
        shuffle (bool): Whether to shuffle dataset or not.

    Returns:
        DataLoader: DataLoader object.
    """
    repeated_data = InfinitelyIndexableDataset(dataset)
    return DataLoader(repeated_data, batch_size=batch_size, shuffle=shuffle, num_workers=2)

