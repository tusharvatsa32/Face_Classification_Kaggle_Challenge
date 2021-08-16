import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import os
from os import listdir
import zipfile
from torchvision import transforms
from PIL import Image 
import natsort

def extract_(file_location,destination):
    with zipfile.ZipFile(file_location,'r') as zip_ref:
        zip_ref.extractall(destination)
    


#Data Augmentation
def data_augmentation(train_data_path,val_data_path,config):
    
    train_dataset=torchvision.datasets.ImageFolder(train_data_path,transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomInvert(),
        transforms.GaussianBlur(config['augmentation']['gaussian_blur_kernel_size']),
        transforms.RandomAffine(config['augmentation']['random_affine_degrees'])   
    ]))
    # transforms.Normalize(config['augmentation']['normalize_mean'],config['augmentation']['normalize_std'])
    val_dataset=torchvision.datasets.ImageFolder(val_data_path,transform= transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomInvert(),
        transforms.GaussianBlur(config['augmentation']['gaussian_blur_kernel_size']),
        transforms.RandomAffine(config['augmentation']['random_affine_degrees']),
       ]))
    # transforms.Normalize(config['augmentation']['normalize_mean'],config['augmentation']['normalize_std']) 
    return train_dataset,val_dataset

def parse_data(datadir):
    img_list = []
    for root, directories, filenames in os.walk(datadir):  
      for filename in filenames:
        filei = os.path.join(root, filename)
        img_list.append(filei)
    return img_list

class ImageDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        # img=torchvision.transforms.Normalize(config['augmentation']['normalize_mean'],config['augmentation']['normalize_std'])(img)
        return img


def inv_mapping(train_dataset):
    a=train_dataset.class_to_idx
    inv_map = {v: k for k, v in a.items()}
    return inv_map









