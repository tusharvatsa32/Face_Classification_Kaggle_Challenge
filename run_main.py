import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import yaml
import os
from torchvision import transforms
from PIL import Image 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import time
import natsort
from model import Model
from datasets import extract_,data_augmentation,inv_mapping,parse_data,ImageDataset
from train import train_epoch,val_epoch
from test import test_out

def run(model,train_data_loader,criterion,optimizer,val_data_loader,scheduler,n_epochs,device):
	loss_train=[]
	for i in range(n_epochs):
		train_loss=train_epoch(model,train_data_loader,criterion,optimizer,device)
		loss_train.append(train_loss)
		dev_loss,acc=val_epoch(model,val_data_loader,criterion,device)
		scheduler.step(acc)

def inference(model,test_loader,criterion,inv_map,device):
    predictions=test_out(model,test_loader,criterion,inv_map,device)
    return predictions

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config_file",default="./config/train_config.yaml")
    parser.add_argument("--inf_config_file",default="./config/inf_config.yaml")
    args = parser.parse_args()
    with open(args.config_file,'r') as f:
        config=yaml.safe_load(f)
    with open(args.inf_config_file,'r') as f:
        inf_config=yaml.safe_load(f)

    extract_(config['data_path'],config['unzip_path'])
    train_dataset,val_dataset=data_augmentation(config['train_data_path'],config['val_data_path'],config)

    train_data_loader=torch.utils.data.DataLoader(train_dataset,batch_size=config['batch_size'],shuffle=config['shuffle'],num_workers=config['num_workers'],drop_last=config['drop_last'])
    
    val_data_loader=torch.utils.data.DataLoader(val_dataset,batch_size=config['batch_size'],shuffle=config['shuffle'],num_workers=config['num_workers'],drop_last=config['drop_last'])
   
    test_dataset=parse_data(inf_config['test_data_path'])
    natsort_data=natsort.natsorted(test_dataset)

    test_data_loader=torch.utils.data.DataLoader(ImageDataset(natsort_data),batch_size=inf_config['batch_size'],shuffle=inf_config['shuffle'],num_workers=inf_config['num_workers'],drop_last=inf_config['drop_last'])

    inv_map=inv_mapping(train_dataset)
    
    cuda=torch.cuda.is_available()
    device=torch.device("cuda" if cuda else "cpu")
    model=Model()
    model.to(device)

    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=config['learning_rate'],momentum=config['momentum'],nesterov=config['nesterov'],weight_decay=config['weight_decay'])
    scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode=config['scheduler']['mode'],patience=config['scheduler']['patience'],threshold=config['scheduler']['threshold'],factor=config['scheduler']['factor'],verbose=config['scheduler']['verbose'])

    run(model,train_data_loader,criterion,optimizer,val_data_loader,scheduler,config['number_epochs'],inv_map,device)
    predictions=inference(model,test_data_loader,criterion,inv_map,device)
    

    






