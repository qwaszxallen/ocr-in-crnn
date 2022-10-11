import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
import string
import pandas as pd
import cv2
import numpy
#!pip install captcha
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.utils.rnn as rnn_utils

characters = "-" + string.digits + "R"
width, height, n_len, n_class = 100, 32, 9, len(characters)

n=1 ## 初始值為1是由於保留0的位置給CTC的空格，所以除了原本37個class外，會加上空格變成38個class
character_index={} ## character對照標籤字典
index_character={} ## 標籤對照character字典
for character in characters:
    character_index[character]=n
    index_character[n]=character
    n+=1 




## CNN Backbone
def make_layers(cfg, batch_norm):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, padding=0)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)]
        elif v == 'S':
            conv2d = nn.Conv2d(512, 512, kernel_size=2, padding=0)
            layers += [conv2d, nn.BatchNorm2d(512), nn.ReLU(inplace=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    return nn.Sequential(*layers)

##定義 Bidirectional GRU

class BiRNN_GRU(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, drop=0.3):
        super(BiRNN_GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.drop=drop
        self.GRU = torch.nn.GRU(input_size,hidden_size, num_layers,batch_first=False,dropout=self.drop,bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection

    def forward(self, x):
        # Forward propagate LSTM
        out, _ = self.GRU(x)# out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out
    
class CRNN(nn.Module):
    def __init__(self, features, num_classes=13, init_weights=True):
        super(CRNN, self).__init__()
        self.features = features
        self.GRU_First=BiRNN_GRU(512,256,2,256,0.3)
        self.GRU_Second=BiRNN_GRU(256,256,2,256,0.3)
        self.output=torch.nn.Linear(256,num_classes) 
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = torch.squeeze(x,2)
        #print(x.shape)
        x = x.permute(2,0,1)
        x = self.GRU_First(x)
        x = self.GRU_Second(x)
        x = self.output(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfgs = { 'backbone': [ 64, 64, 'M', 128, 128, 'M', 256, 256, 'C', 512, 512, 'C', 'S'],}
#(32,128,3),(32,128,64)*2,(16,64,64),(16,64,128)*2,(8,32,128),(8,32,256)*2,(4,16,256),(4,16,512)*2,(2,15,512),(1,14,512)
def CRNN_BN(config,batch_norm = True, **kwargs):

    return CRNN(make_layers(config['backbone'], batch_norm=batch_norm), **kwargs)
  
model = CRNN_BN(cfgs) 
if(os.path.exists('withblank5.pt')):
    print('use checkpoint...')
    checkpoint = torch.load('withblank5.pt')
    model.load_state_dict(checkpoint['model'])
   

for img in os.listdir("./22"):
    test = cv2.imread("./22/"+img)
    test = cv2.cvtColor(test,cv2.COLOR_BGR2RGB)
    #test = cv2.resize(test,(100,32))
    img_input = np.array(test)

    img_input=Variable(torch.tensor(img_input).type(torch.FloatTensor).unsqueeze(0)).permute(0,3,1,2)
    
    output=model(img_input)
    output=F.softmax(output,2)
    print(output.squeeze(1).argmax(1))
    word=''
    n=0
    ## 其中0代表預測為空格，如果預測相同字符之間沒有空格要移除
    for word_result in output.squeeze(1).argmax(1):
        character_index=word_result.item()
        if (n != character_index):
            if (character_index > 0):            
                word += index_character[character_index]
            n = character_index 

    print(word) 
    while(len(word) > 9):
        word = word[:-1]

    plt.title(word)
    plt.imshow(test)
    plt.show()