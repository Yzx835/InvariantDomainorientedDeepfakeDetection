import os
import numpy as np
import torch
import albumentations as A
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch.functional import img_to_tensor
import random
import json
import copy
from my_transforms import region_erasing_1, region_erasing_2

random.seed(233)

class MyDataset_FF(Dataset): 
    def __init__(self, data_dir, split_path, region_erase_path1=None, region_erase_path2=None,
                 data_type=['raw', 'deepfakes', 'faceswap', 'face2face','neuraltextures'], 
                 train_val_test='all', normalize=None, transforms=None, oversample_real=True, erase_rate=0):
        if normalize == None:
            self.normalize={"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
        else:
            self.normalize=normalize
            
        if transforms == None:
            self.transforms = A.Compose([
                    A.Resize(height=224, width=224),
                ])
        else:
            self.transforms = transforms
        
        if train_val_test=='train':
            self.ff_split = os.path.join(split_path,'train.json')
        elif train_val_test=='val':
            self.ff_split = os.path.join(split_path,'val.json')
        elif train_val_test=='test':
            self.ff_split = os.path.join(split_path,'test.json')

        with open(self.ff_split,'r') as f:
            self.ff_split = json.load(f)
        self.ff_dict = set()
        for item in self.ff_split:
            self.ff_dict.add(item[0])
            self.ff_dict.add(item[1])
        self.ff_length = 0

        self.train_val_test = train_val_test
        self.oversample_real = oversample_real and train_val_test=='train'
        self.erase_rate = erase_rate
        
        self.datas = {}
        self.all_datas = []
        self.types = {0:[], 1:[]}
        for d_type in data_type:
            self.datas[d_type] = []
            d_dir = os.path.join(data_dir, d_type)
            files = os.listdir(d_dir)
            for file in files:
                containId = file.split('_')
                if containId[0] not in self.ff_dict: 
                    continue
                self.ff_length +=1
                images = os.listdir(os.path.join(d_dir, file))
                label = 0 if d_type == 'raw' else 1
                for image in images:
                    self.types[label].append(os.path.join(d_dir, file, image))
                    self.datas[d_type].append((os.path.join(d_dir, file, image), label))
                    self.all_datas.append((os.path.join(d_dir, file, image), label, d_type))
        print("TOTAL FF++", len(self.all_datas), 'REAL', len(self.types[0]), 'FAKE', len(self.types[1]))
        print("file_dir count", self.ff_length)
        
        if self.erase_rate > 0:
            with open(region_erase_path1, 'r') as f:
                self.landmarks_5 = json.load(f)
            with open(region_erase_path2, 'r') as f:
                self.landmarks_68 = json.load(f)
    
    def get_data(self):
        return self.all_datas
    
    def set_data(self, datas):
        self.all_datas = datas
    
    def reset(self, seed=233):
        if self.oversample_real:
            self.all_datas = []
            len_fake = len(self.types[1])
            random.shuffle(self.types[1])
            while(len(self.all_datas) < len_fake):
                for image in self.types[0]:
                    self.all_datas.append((image, 0))
            for image in self.types[1]:
                self.all_datas.append((image, 1))
    
    def rate(self, real, fake):
        len_real = len(self.types[0])
        len_fake = len(self.types[1])
        self.all_datas = []
        print('REAL', int(real * len_real), 'FAKE', int(fake * len_fake))
        i = 0
        for _ in range(int(real * len_real)):
            if i == len(self.types[0]):
                i = 0
            self.all_datas.append((self.types[0][i], 0))
            i += 1
        i = 0
        for _ in range(int(fake * len_fake)):
            if i == len(self.types[1]):
                i = 0
            self.all_datas.append((self.types[1][i], 1))
            i += 1
    
    def __len__(self):
        return len(self.all_datas)
    
    def __getitem__(self, index):
        
        
        img_path, label = self.all_datas[index][:2]
        
        img = Image.open(img_path)
        img = np.array(img)
        
        height, width = img.shape[:2]
        
        if self.train_val_test == 'train' and random.random() < self.erase_rate:
            if random.random() < 0.8:
                
                if img_path in self.landmarks_5.keys():
                    landmarks = copy.deepcopy(self.landmarks_5[img_path])
                    for i in range(len(landmarks)):
                        landmarks[i][0] = int(landmarks[i][0] * height)
                        landmarks[i][1] = int(landmarks[i][1] * width)
                        if landmarks[i][0] < 0:
                            landmarks[i][0] = 0
                        if landmarks[i][0] >= height:
                            landmarks[i][0] = height - 1
                        if landmarks[i][1] < 0:
                            landmarks[i][1] = 0
                        if landmarks[i][1] >= width:
                            landmarks[i][1] = width - 1
                    img, mask = region_erasing_1(img, landmarks)
            else:
                
                if img_path in self.landmarks_68.keys():
                    landmarks = copy.deepcopy(self.landmarks_68[img_path])
                    for i in range(len(landmarks)):
                        landmarks[i][0] = int(landmarks[i][0] * height)
                        landmarks[i][1] = int(landmarks[i][1] * width)
                        if landmarks[i][0] < 0:
                            landmarks[i][0] = 0
                        if landmarks[i][0] >= height:
                            landmarks[i][0] = height - 1
                        if landmarks[i][1] < 0:
                            landmarks[i][1] = 0
                        if landmarks[i][1] >= width:
                            landmarks[i][1] = width - 1   
                    img, mask = region_erasing_2(img, landmarks)
            if index % 100 == 0:
                Image.fromarray(img).save('image_landmarks.png')
        data = self.transforms(image=img)
        
        img = data['image']
        if index % 100 == 0:
            Image.fromarray(img).save('image_transformed.png')
        img = img_to_tensor(img, self.normalize)
        return img, label    

class MyDataloader:
    def __init__(self, loaders):
        self.loaders = loaders
        self.loader_iters = []
        for loader in self.loaders:
            self.loader_iters.append(iter(loader))
    
    def __len__(self):
        _max = 0
        for loader in self.loaders:
            _max = max(_max, len(loader))
        return _max
    
    def get_next(self):
        images = []
        labels = []
        for i  in range(len(self.loader_iters)):
            try:
                img, label = self.loader_iters[i].next()
            except StopIteration:
                self.loader_iters[i] = iter(self.loaders[i])
                img, label = self.loader_iters[i].next()
            images.append(img)
            labels.append(label)
        images = torch.cat(images, 0)
        labels = torch.cat(labels, 0)
        return images, labels