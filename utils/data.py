import os

import numpy as np
from torch.utils.data import DataLoader,Dataset

class MultiChannelData(Dataset):
    def __init__(self,path,size,channel):
        self.path = path
        self.size = size
        self.channel = channel
        self.files = os.listdir(path)

    def __getitem__(self, item):
        x = np.fromfile(self.path+'/'+self.files[item],dtype=np.float32).reshape(self.channel,self.size,self.size)
        y = np.fromfile(self.path+'/../label/'+self.files[item],dtype=np.float32).reshape(1,self.size,self.size)
        return x,y,self.files[item]

    def __len__(self):
        return len(self.files)

