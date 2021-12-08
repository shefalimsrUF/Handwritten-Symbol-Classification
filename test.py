#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessory libraries
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
get_ipython().run_line_magic('matplotlib', 'inline')

from PIL import Image as im
import numpy as np
import os


# # Transforming Data

# In[2]:




image_path = './final_data/Train_Images.npy'
label_path = './final_data/Train_Labels.npy'

# ------------------- downloading the model -----------------------

import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

file_id = '14LOIRDuM1xcmEm89EbXsdmj6KUtf-Xjj'

destination = 'model.pth'
download_file_from_google_drive(file_id, destination)


# -------------------  testing --------------------


X = np.load(image_path)
y = np.load(label_path)


# In[3]:

os.makedirs('./images')
for i in range(1,26,1):
    os.makedirs('./images/{}'.format(i))
    
count = 0
for clss,img in zip(y,X):
    data = im.fromarray(img)
    data.save('./images/{}/img_{}.jpeg'.format(clss+1,count))
    count+=1

image_dir = './images/'
dataset = ImageFolder(image_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))


# # # Loading Data

# # In[6]:


data_dir = "./images/" 


# In[7]:


dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))


# In[12]:


batch_size = 256

train_dl = DataLoader(dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)


# # # Model

# # In[25]:
class ImageClassificationBase(nn.Module):
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss    
   
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class SymbolClassification(ImageClassificationBase):
    
    def __init__(self):
        
        super().__init__()
        self.network = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(32,64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        
            nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(128 ,128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.Conv2d(256,256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            
            
            nn.Flatten(),
            nn.Linear(82944,1024),
            nn.ReLU(),
            nn.Linear(1024, 25),
            nn.Softmax()
            
        )
    
    def forward(self, xb):
        return self.network(xb)
    

PATH = 'model.pth'
# model = torch.load(PATH)
# model.eval()

model = SymbolClassification()
model.load_state_dict(torch.load(PATH))
model.eval()

# In[26]:


# torch.save(model.state_dict(), 'symbol_classification_state_400.pth')


# # GPU

# In[20]:


def get_default_device():
    """ Set Device to GPU or CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    

def to_device(data, device):
    "Move data to the device"
    if isinstance(data,(list,tuple)):
        return [to_device(x,device) for x in data]
    return data.to(device,non_blocking = True)

class DeviceDataLoader():
    """ Wrap a dataloader to move data to a device """
    
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        """ Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b,self.device)
            
    def __len__(self):
        """ Number of batches """
        return len(self.dl)
    
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


# In[21]:


device = get_default_device()
# device


# In[24]:


batch_size = 256
val_loader = DeviceDataLoader(DataLoader(dataset, batch_size*2), device)
to_device(model, device)
result = evaluate(model, val_loader)
print('Val data accuracy - ', result['val_acc']*100)
print('Val data loss - ', result['val_loss'])