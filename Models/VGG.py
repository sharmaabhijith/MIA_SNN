from unicodedata import numeric
import torch.nn as nn
import numpy as np
cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ]
}

class Small(nn.Module):
    def __init__(self, vgg_name, num_classes,in_channel):
        super(Small,self).__init__()
        self.init_channels = in_channel
        print(in_channel)
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32768, 10,bias=True),
                #nn.ReLU(inplace=True),
                #nn.Linear(4096, 4096,bias=False),
                #nn.ReLU(inplace=True),
                #nn.Linear(4096, num_classes,bias=False)
            )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
    
    
    def _make_layers(self, cfg):
        layers = []
        
        layers.append(nn.Conv2d(self.init_channels, 32, kernel_size=3, padding=1,bias=True))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace=True))
        #layers.append(nn.Dropout(dropout))
        self.init_channels = 32
        layers.append(nn.Conv2d(self.init_channels, 32, kernel_size=3, padding=1,bias=True))
        #layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU(inplace=True))
        #layers.append(nn.Dropout(dropout))
        self.init_channels = 32
        return nn.Sequential(*layers)
    def forward(self, x,thresholds=0,L=0):
        out = x
        #print(out[0,0])
        counter = 0
        for i in range(len(self.layer1)):
            out = self.layer1[i](out)
            #if i==1 or i==3:
                #print(out.sum())
            if 'ReLU' in str(self.layer1[i]):
                
                counter += 1    
                if counter == L:
                    return out
        for i in range(len(self.classifier)):
                #print(out.sum())
                out = self.classifier[i](out)
                #if i==2 or i==5:
                #    print(out.sum())
            
                #print(out.sum())
                if 'ReLU' in str(self.classifier[i]):
                    #print(out.sum())
                    counter += 1    
                    if counter == L:
                        return out
        
        return out

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout,in_channel):
        super(VGG, self).__init__()
        self.init_channels = in_channel
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        
        if num_classes == 1000:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512*7*7, 4096,bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096,bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes,bias=True)
            )
        elif num_classes == 200:
            print("here..............")
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(2048, 4096), #2048
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 4096,bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, 4096,bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(4096, num_classes,bias=True)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)
           
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1,bias=True))
                layers.append(nn.BatchNorm2d(x,track_running_stats=True,affine=True))
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
                self.init_channels = x
        return nn.Sequential(*layers)
    

    def forward(self, x, thresholds=0, L=0, t=0, prev_L=0):
        out = x
        #print(out[0,0])
        counter = 0
        layer_count = 0
        features = {}
        #print(L)
        for i in range(len(self.layer1)):
            if 'SPIKE_layer' in str(self.layer1[i]):
                if prev_L <= counter:
                    out = self.layer1[i](out, t)
                #print("spike",counter+1,out.sum())
                counter += 1
                if counter == L:
                    return out
            else:
                if prev_L <= counter:
                    out = self.layer1[i](out)
            #out.sum(-1).detach().cpu()
            layer_count += 1
            #print(i,out.sum())
            #if i==1:
                #return out
            if 'LIFSpike' in str(self.layer1[i]):
                #print("spike",counter,out.sum())
                
                counter += 1    
                if counter == L:
                    return out
            if 'ReLU' in str(self.layer1[i]):
                #print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
        #return
            #out = self.layer1[i](out)

        for i in range(len(self.layer2)):
            if 'SPIKE_layer' in str(self.layer2[i]):
                if prev_L <= counter:
                    out = self.layer2[i](out, t)
                #print("spike",counter+1,out.sum())
                counter += 1
                if counter == L:
                    return out
            else:
                if prev_L <= counter:
                    out = self.layer2[i](out)
            
            #out.sum(-1).detach().cpu()
            layer_count += 1
            
            #print(i,out.sum())
            if 'LIFSpike' in str(self.layer2[i]):
                #print("spike",counter,out.sum())
                counter += 1 
                if counter == L:
                    return out
            if 'ReLU' in str(self.layer2[i]):
                #print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
        
        for i in range(len(self.layer3)):
            if 'SPIKE_layer' in str(self.layer3[i]):
                if prev_L <= counter:
                    out = self.layer3[i](out, t)
                #print("spike",counter+1,out.sum())
                counter += 1
                if counter == L:
                    return out
            else:
                if prev_L <= counter:
                    out = self.layer3[i](out)
            #out.sum(-1).detach().cpu()
            layer_count += 1
            
            if 'LIFSpike' in str(self.layer3[i]):
                #print("spike",counter,out.sum())
                counter += 1 
                if counter == L:
                    return out
            if 'ReLU' in str(self.layer3[i]):
                #print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
            #out = self.layer1[i](out)

        for i in range(len(self.layer4)):
            if 'SPIKE_layer' in str(self.layer4[i]):
                
                if prev_L <= counter:
                    out = self.layer4[i](out, t)
                #print("spike",counter+1,out.sum())
                counter += 1
                if counter == L:
                    return out
            else:
                if prev_L <= counter:
                    out = self.layer4[i](out)
            #out.sum(-1).detach().cpu()
            layer_count += 1
            
            if 'LIFSpike' in str(self.layer4[i]):
                #print("spike",counter,out.sum())
                counter += 1 
                if counter == L:
                    return out
            if 'ReLU' in str(self.layer4[i]):
                #print("relu",counter,out.sum())
                #if counter +1 ==13:
                #    return out
                counter += 1    
                if counter == L:
                    return out
       
        for i in range(len(self.layer5)):
            if 'SPIKE_layer' in str(self.layer5[i]):
                # if counter +1 ==13:
                #     print("bias",self.layer5[i-1].bias[0:2])
                #     return out
                if prev_L <= counter:
                    out = self.layer5[i](out, t)
                #print("spike",counter+1,out.sum())
                counter += 1
                if counter == L:
                    return out
            else:
                if prev_L <= counter:
                    out = self.layer5[i](out)
                # if counter +1 ==13 and  'Conv2d' in str(self.layer5[i]):
                #     #np.save('./features/weights.npy',self.layer5[i].weight.detach().cpu().numpy())
                #     #np.save('./features/bias.npy',self.layer5[i].bias.detach().cpu().numpy())
                #     #print("bias",self.layer5[i].weight.shape,self.layer5[i].bias.shape)
                #     print(str(self.layer5[i]))
                #     return out
                
            #out.sum(-1).detach().cpu()
            layer_count += 1
            
            ##print(out.sum())
            if 'LIFSpike' in str(self.layer5[i]):
                #print("spike",counter,out.sum())
                counter += 1 
                if counter == L:
                    return out
            if 'ReLU' in str(self.layer5[i]):
                #print("relu",counter,out.sum())
                
                counter += 1    
                if counter == L:
                    return out
        for i in range(len(self.classifier)):
            ##print(out.shape)
            #out.sum(-1).detach().cpu()
            layer_count += 1
            
            if 'SPIKE_layer' in str(self.classifier[i]):
                if prev_L <= counter:
                    out = self.classifier[i](out, t)
                #print("spike",counter+1,out.sum())
                counter += 1
                if counter == L:
                    return out
            else:
                if prev_L <= counter:
                    out = self.classifier[i](out)
            if 'LIFSpike' in str(self.classifier[i]):
                #print("spike",counter,out.sum())
                counter += 1 
                if counter == L:
                    return out
            
            
            ##print(out.sum())
            if 'ReLU' in str(self.classifier[i]):
                #print("relu",counter,out.sum())
                counter += 1    
                if counter == L:
                    return out
              
        #return features
        return out


class VGG_normed(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout):
        super(VGG_normed, self).__init__()
        self.num_classes = num_classes
        self.module_list = self._make_layers(cfg[vgg_name], dropout)


    def _make_layers(self, cfg, dropout):
        layers = []
        for i in range(5):
            for x in cfg[i]:
                if x == 'M':
                    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    layers.append(nn.Conv2d(3, x, kernel_size=3, padding=1))
                    layers.append(nn.ReLU(inplace=True))
                    layers.append(nn.Dropout(dropout))
                    self.init_channels = x
        layers.append(nn.Flatten())
        if self.num_classes == 1000:
            layers.append(nn.Linear(512*7*7, 4096))
        else:
            layers.append(nn.Linear(512, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, 4096))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4096, self.num_classes))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.module_list(x)

class CifarNet(nn.Module):  # Example net for CIFAR10
    def __init__(self):
        super(CifarNet, self).__init__()
        #self.conv0 = nn.Conv2d(1, 1, 5, 2)
        self.conv0 = nn.Conv2d(3, 128, 3, 1, 1)
        self.bn0 = nn.BatchNorm2d(128)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(256, 512, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(512)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(2)
        self.conv3 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(1024, 512, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(8 * 8 * 512, 1024)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1024, 512)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(512, 10)

        
    def forward(self, x,thresholds=0,L=0,t=0):
        counter = 0
        
        x = self.conv0(x)
        x = self.bn0(x)
        
        if 'SPIKE_layer' in str(self.relu0):
            x = self.relu0(x,t)
            counter += 1 
            if counter == L:
                return x
        if 'ReLU' in str(self.relu0):
            x = self.relu0(x)
            counter += 1    
            if counter == L:
               return x
        x = self.conv1(x)
        x = self.bn1(x)
        
        if 'SPIKE_layer' in str(self.relu1):
            counter += 1 
            x = self.relu1(x,t)
            if counter == L:
                return x
        if 'ReLU' in str(self.relu1):
            x = self.relu1(x)
            counter += 1    
            if counter == L:
               return x
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        if 'SPIKE_layer' in str(self.relu2):
            counter += 1 
            x = self.relu2(x,t)
            if counter == L:
                return x
        if 'ReLU' in str(self.relu2):
            x = self.relu2(x)
            counter += 1    
            if counter == L:
               return x
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        if 'SPIKE_layer' in str(self.relu3):
            counter += 1 
            x = self.relu3(x,t)
            if counter == L:
                return x
        if 'ReLU' in str(self.relu3):
            counter += 1    
            x = self.relu3(x)
            if counter == L:
               return x
        x = self.conv4(x)
        x = self.bn4(x)
        
        if 'SPIKE_layer' in str(self.relu4):
            x = self.relu4(x,t)
            counter += 1 
            if counter == L:
                return x
        if 'ReLU' in str(self.relu4):
            x = self.relu4(x)
            counter += 1    
            if counter == L:
               return x
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.fc1(x)
        
        if 'SPIKE_layer' in str(self.relu5):
            x = self.relu5(x,t)
            counter += 1 
            if counter == L:
                return x
        if 'ReLU' in str(self.relu5):
            counter += 1    
            x = self.relu5(x)
            if counter == L:
               return x
        x = self.fc2(x)
        
        if 'SPIKE_layer' in str(self.relu6):
            x = self.relu6(x,t)
            counter += 1 
            if counter == L:
                return x
        if 'ReLU' in str(self.relu6):
            x = self.relu6(x)
            counter += 1    
            if counter == L:
               return x
        x = self.fc3(x)
        return x

def vgg11(num_classes=10, dropout=0,in_channel=3, **kargs):
    return VGG('VGG11', num_classes, dropout,in_channel)


def vgg13(num_classes=10, dropout=0,in_channel=3, **kargs):
    return VGG('VGG13', num_classes, dropout,in_channel)


def vgg16(num_classes=10, dropout=0,in_channel=3, **kargs):
    return VGG('VGG16', num_classes, dropout,in_channel)


def vgg19(num_classes=10, dropout=0, **kargs):
    return VGG('VGG19', num_classes, dropout)


def small(num_classes=10, dropout=0,in_channel=3, **kargs):
    return Small('VGG19', num_classes,in_channel)


def vgg16_normed(num_classes=10, dropout=0, **kargs):
    return VGG_normed('VGG16', num_classes, dropout)

def cifarnet():
    return CifarNet()
