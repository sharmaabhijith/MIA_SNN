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
    train: bool,
) -> DataLoader:
    """
    Function to get DataLoader.

    Args:
        dataset (torchvision.datasets): The whole dataset.
        batch_size (int): Batch size for getting signals.
        train (bool): Whether loader is for training or testing
    Returns:
        DataLoader: DataLoader object.
    """
    repeated_data = InfinitelyIndexableDataset(dataset)
    if train:
        train_transformed_data = TransformDataset(repeated_data, train)
        return DataLoader(train_transformed_data, batch_size=batch_size, shuffle=True, num_workers=2)
    else:
        test_transformed_data = TransformDataset(repeated_data, train)
        return DataLoader(test_transformed_data, batch_size=batch_size, shuffle=False, num_workers=2)
