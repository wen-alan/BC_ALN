# -*- coding:utf-8 -*-
import numpy as np
import os
import torch
from PIL import Image
import torch.utils.data as data

# Custom dataset
class MyDataset(data.Dataset):
    def __init__(self, split='train', data_dir=None, transform=None, ratio=0.8):
        self.split = split
        img_list = os.listdir(data_dir)
        trainset_count = int(len(img_list)*ratio)
        np.random.seed(0)
        np.random.shuffle(img_list)

        if self.split == 'train':
            train_img_list = img_list[:trainset_count]
            train_img_label = [int(img_name.split('_')[-1].split('.')[0]) for img_name in train_img_list]
            imgs = zip(train_img_list, train_img_label)
            # with open('data/train_set_cnn.csv', 'w') as f_tr:
            #     f_tr.write('patient_name \n')
            # with open('data/train_set_cnn.csv', 'a') as f_tr:
            #     train_img_name = [img_name.split('_')[0] for img_name in train_img_list]
            #     for tr_name in train_img_name: f_tr.write('%s\n' %(tr_name))
        elif self.split == 'train_ord':
            train_img_list = img_list[:trainset_count]
            train_img_label = [int(img_name.split('_')[-1].split('.')[0]) for img_name in train_img_list]
            imgs = zip(train_img_list, train_img_label)
            # with open('data/train_set_cnn_ord.csv', 'w') as f_tr: f_tr.write('patient_name \n')
            # with open('data/train_set_cnn_ord.csv', 'a') as f_tr:
            #     train_img_name = [img_name.split('_')[0] for img_name in train_img_list]
            #     for tr_name in train_img_name: f_tr.write('%s\n' %(tr_name))
        elif self.split == 'test':
            test_img_list = img_list[trainset_count:]
            test_img_label = [int(img_name.split('_')[-1].split('.')[0]) for img_name in test_img_list]
            imgs = zip(test_img_list, test_img_label)
            # with open('data/test_set_cnn.csv', 'w') as f_te: f_te.write('patient_name \n')
            # with open('data/test_set_cnn.csv', 'a') as f_te:
            #     test_img_name = [img_name.split('_')[0] for img_name in test_img_list]
            #     for te_name in test_img_name: f_te.write('%s\n' %(te_name))
        elif self.split == 'val':
            val_img_label = [int(img_name.split('_')[-1].split('.')[0]) for img_name in img_list]
            imgs = zip(img_list, val_img_label)
            # with open('data/val_set_cnn.csv', 'w') as f_val: f_val.write('patient_name \n')
            # with open('data/val_set_cnn.csv', 'a') as f_val:
            #     val_img_name = [img_name.split('_')[0] for img_name in img_list]
            #     for val_name in val_img_name: f_val.write('%s\n' %(val_name))
        else:
            print('Error input, if shuld be train or test!')
            exit(0)

        self.imgs = list(imgs)
        self.data_dir = data_dir
        self.transforms = transform

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        torch.set_printoptions(profile="full")
        # print("img1",Image.open(self.data_dir + '/'+img_name).load()[20,20])
        img = Image.open(self.data_dir + '/'+img_name).convert('RGB') #convert('L')
        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

