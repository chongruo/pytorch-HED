import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import random


import pdb



class BSD500Dataset():
    def __init__(self, cfg):

        self.cfg = cfg
        self.rootdir = cfg.DATA.root
        self.train_list = cfg.DATA.train_list  
        
        ### data 
        self.all_path_list = []
        with open('/'.join([self.rootdir, self.train_list]), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                cur_pair = line.split(' ')
                
                self.all_path_list.append( cur_pair )
        print('in data_loader: Train data preparation done')

        '''
        ### transformer
        mean = [float(item) / 255.0 for item in cfg.DATA.mean]
        std = [1,1,1]
        
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std)
                    ])
            

        self.targetTransform = transforms.Compose([
                            transforms.ToTensor()
                          ])
        '''

    def mytransfrom(self, img, gt):
        '''
        input:  img,gt, PIL image
        output: tensor
        '''


        ### ColorJitterUG:
        if self.cfg.DATA.AUG.ColorJitter:
            color_jitter = transforms.ColorJitter(brightness = self.cfg.DATA.AUG.brightness,
                                                  contrast = self.cfg.DATA.AUG.contrast,
                                                  saturation = self.cfg.DATA.AUG.saturation,
                                                  hue = self.cfg.DATA.AUG.hue )
            color_jitter_transform = color_jitter.get_params(color_jitter.brightness, color_jitter.contrast,
                                                             color_jitter.saturation, color_jitter.hue)
            img = color_jitter_transform(img)
        

        
        if self.cfg.DATA.AUG.HFlip:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt = F.hflip(gt)


        ### ToTensor
        img = F.to_tensor(img)
        gt = F.to_tensor(gt)

        ### Normalization
        mean = [float(item) / 255.0 for item in self.cfg.DATA.mean]
        std = [1,1,1]

        normalizer = transforms.Normalize(mean=mean, std=std)
        img = normalizer(img)   
    
        return img, gt
        

    def __getitem__(self, idx):
        img_path, gt_path = [ '/'.join([self.rootdir, item]) for item in self.all_path_list[idx] ]

        img = Image.open(img_path).convert('RGB')
        gt  = Image.open(gt_path).convert('L')


        img_t, gt_t = self.mytransfrom(img, gt)

        if self.cfg.DATA.gt_mode=='gt_half':
            gt_t[gt_t>=0.5] = 1 
            gt_t[gt_t<0.5] = 0
        
        
        return img_t, gt_t

    
    def __len__(self):
        return len(self.all_path_list)


 




####################################################################################################

class BSD500DatasetTest():
    def __init__(self, cfg):
        self.rootdir = cfg.DATA.root
        self.train_list = cfg.DATA.test_list  
        
        ### data 
        self.all_path_list = []
        with open('/'.join([self.rootdir, self.train_list]), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line[:-1]
                self.all_path_list.append( line )
        print('in data_loader: Test data preparation done')

        ### transformer
        mean = [float(item) / 255.0 for item in cfg.DATA.mean]
        std = [1,1,1]
        self.transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean,std)
                    ])
        

    def __getitem__(self, idx):
        img_path = '/'.join([self.rootdir, self.all_path_list[idx]])
        img_filename = img_path.split('/')[-1].split('.')[0] 

        img = Image.open(img_path).convert('RGB')
        img_t = self.transform(img)
        
        
        return (img_t, img_filename)

    
    def __len__(self):
        return len(self.all_path_list)



