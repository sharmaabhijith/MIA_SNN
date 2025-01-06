from .getdataloader import *

def datapool(DATANAME, batchsize, num_workers, train_test_split=0.9, shuffle=True):
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
