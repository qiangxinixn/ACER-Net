import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
import h5py
import glob

BS = opt.bs
print(BS)
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size

class RESIDE_Dataset(data.Dataset):
    def __init__(self,path,train,size=crop_size,format='.png'):
        print(f'path={path}')
        super(RESIDE_Dataset,self).__init__()
        self.size = size
        print('crop size',size)
        self.train=train

        self.h5path = path # 输入h5文件
        self.file_list=os.listdir(path)
    def __getitem__(self, index):

        file_name = os.path.join(self.h5path,self.file_list[index])
        f = h5py.File(file_name, 'r')
        
        haze = f['data']
        clear = f['label']
        
        i=random.randint(0,haze.shape[0])   
                
        haze = haze[i]
        clear = clear[i]
        
        # we might get out of bounds due to noise
        haze = np.clip(haze, 0, 1)
        # we might get out of bounds due to noise
        clear = np.clip(clear, 0, 1)
        haze = np.asarray(haze, np.float32)
        clear = np.asarray(clear, np.float32)
        
        
        # haze = Image.fromarray(np.array(haze))
        # clear = Image.fromarray(np.array(clear))
        #print(f'haze={haze}')
        #exit()
        # clear = tfs.CenterCrop(haze.size[::-1])(clear)

        # if not isinstance(self.size, str):
        #     i, j, h, w = tfs.RandomCrop.get_params(haze,output_size=(self.size,self.size))
        #     haze = FF.crop(haze, i, j, h, w)
        #     clear = FF.crop(clear, i, j, h, w)

        # haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        #print(f'haze={haze.shape}')
        #exit()
        return haze, clear

    def augData(self, data, target):

        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90*rand_rot)
                target = FF.rotate(target, 90*rand_rot)

        data = tfs.ToTensor()(data)
        #data = tfs.Normalize(mean=[0.64, 0.6, 0.58],std=[0.14,0.15, 0.152])(data) # 加上归一化
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.file_list)

root = '/root/autodl-tmp/xx/AECR-Net/data/'
_ITS_train_loader=RESIDE_Dataset(root, train=True,size=crop_size)
ITS_train_loader=DataLoader(_ITS_train_loader,batch_size=BS,shuffle=False)
# ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(root+'ITS_train/', train=True,size=crop_size),batch_size=BS,shuffle=True,num_samples=1)
ITS_test_loader=DataLoader(dataset=RESIDE_Dataset(root,train=False,size='whole img'),batch_size=1,shuffle=False)
ITS_train_loader_whole=DataLoader(dataset=RESIDE_Dataset(root,train=False,size='whole img'),batch_size=BS,shuffle=False)
rebuttal_test_loader = DataLoader(dataset=RESIDE_Dataset(root,train=False,size='whole img'),batch_size=100,shuffle=False)
# for debug
#ITS_test = '/home/why/workspace/CDNet/net/debug/test_h5/'
#ITS_test_loader_debug=DataLoader(dataset=RESIDE_Dataset(ITS_test,train=False,size='whole img'),batch_size=1,shuffle=False)
for x,y in _ITS_train_loader:
    print(x)
    print(y)
    
if __name__ == "__main__":
    pass
