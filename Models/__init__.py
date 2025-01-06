from .ResNet import *
from .VGG import *

def modelpool(MODELNAME, DATANAME):
    in_channel=3
    if 'tiny-imagenet' in DATANAME.lower():
        num_classes = 200
       
    elif 'imagenet' in DATANAME.lower():
        num_classes = 1000
    elif '100' in DATANAME.lower():
        num_classes = 100
    else:
        num_classes = 10
    if 'fashion' in DATANAME.lower():
        in_channel=1
    if MODELNAME.lower() == 'vgg16':
        return vgg16(num_classes=num_classes,in_channel=in_channel)
    elif MODELNAME.lower() == 'vgg11':
        return vgg11(num_classes=num_classes,in_channel=in_channel)
    if MODELNAME.lower() == 'vgg13':
        return vgg13(num_classes=num_classes,in_channel=in_channel)
    elif MODELNAME.lower() == 'resnet18':
        return resnet18(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet34':
        return resnet34(num_classes=num_classes)
    elif MODELNAME.lower() == 'resnet20':
        return resnet20(num_classes=num_classes)
    elif MODELNAME.lower() == 'small':
        return small(num_classes=num_classes,in_channel = in_channel)
    elif MODELNAME.lower() == 'cifarnet':
        return cifarnet()
    else:
        print("still not support this model")
        exit(0)
