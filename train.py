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
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
import pandas as pd
import itertools




# # Image from array data 
import numpy as np
X_train = np.load('./final_data/Train_Images.npy')
y_train = np.load('./final_data/Train_Labels.npy')

# #Splitting the data into training and test set
X_train, X_test, y_train, y_test = train_test_split( X_train, y_train, test_size=0.2, random_state=3)
X_train, X_val, y_train, y_val = train_test_split( X_train, y_train, test_size=0.2, random_state=3)
print(X_train.shape,X_test.shape,X_val.shape)

for i in range(1,26,1):
    os.makedirs('./image_data/train/{}'.format(i))

for i in range(1,26,1):
    os.makedirs('./image_data/test/{}'.format(i))
    
for i in range(1,26,1):
    os.makedirs('./image_data/val/{}'.format(i))
    
count = 0
for clss,img in zip(y_train,X_train):
    data = im.fromarray(img)
    data.save('./image_data/train/{}/train_{}.jpeg'.format(clss+1,count))
    count+=1
    
count = 0
for clss,img in zip(y_test,X_test):
    data = im.fromarray(img)
    data.save('./image_data/test/{}/test_{}.jpeg'.format(clss+1,count))
    count+=1
    
count = 0
for clss,img in zip(y_val,X_val):
    data = im.fromarray(img)
    data.save('./image_data/val/{}/val_{}.jpeg'.format(clss+1,count))
    count+=1


# In[2]:


data_dir = "./image_data/train/" 
test_data_dir = "./image_data/test/"
val_data_dir = "./image_data/val/"


# In[3]:


dataset = ImageFolder(data_dir,transform = transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
test_dataset = ImageFolder(test_data_dir,transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))
val_dataset = ImageFolder(val_data_dir,transforms.Compose([
    transforms.Resize((150,150)),transforms.ToTensor()
]))


# In[4]:


img, label = dataset[1]
print(img.shape,label)


# In[5]:


print(f"Images in training data : {len(dataset)}")
print(f"Images in test data : {len(test_dataset)}")
print(f"Images in val data : {len(val_dataset)}")


# In[6]:


print("Follwing classes are there : \n",dataset.classes)


# # Loading data

# In[7]:


batch_size = 256

train_dl = DataLoader(dataset, batch_size, shuffle = True, num_workers = 4, pin_memory = True)
val_dl = DataLoader(val_dataset, batch_size*2, num_workers = 4, pin_memory = True)


# In[10]:


# from torchvision.utils import make_grid

# def show_batch(dl):
#     for images, labels in dl:
#         fig,ax = plt.subplots(figsize = (16,12))
#         ax.set_xticks([])
#         ax.set_yticks([])
#         ax.imshow(make_grid(images,nrow=16).permute(1,2,0))
#         break

# show_batch(train_dl)


# # Base Model for Image Classification:

# In[9]:


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


# # Symbol Classfication Model

# In[10]:


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


# In[11]:


model = SymbolClassification() 
# model


# In[12]:


# for images, labels in train_dl:
#     print('images.shape:', images.shape)
#     out = model(images)
#     print('out.shape:', out.shape)
#     print('out[0]:', out[0])
#     break


# ## Helper Function or classes to Load Data into GPU

# In[13]:


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


# In[14]:


device = get_default_device()
# device


# In[15]:


# load the into GPU
train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)


# # Model Fitting

# In[16]:


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(),lr)
    for epoch in range(epochs):
        
        model.train()
        train_losses = []
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    
    return history


# In[17]:


#load the model to the device
model = to_device(SymbolClassification(),device)


# In[18]:


#initial evaluation of the model
# evaluate(model,val_dl)


# In[19]:


#set the no. of epochs, optimizer funtion and learning rate
num_epochs = 400
opt_func = torch.optim.Adam
lr = 0.0001

#lr = 0.001


# In[20]:


#fitting the model on training data and record the result after each epoch
history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)


# In[35]:


# Apply the model on test dataset and Get the results
test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
result = evaluate(model, test_loader)
print('Test data accuracy - ', result['val_acc']*100)
print('Test data loss - ', result['val_loss'])


# In[36]:


val_loader = DeviceDataLoader(DataLoader(val_dataset, batch_size*2), device)
result = evaluate(model, val_loader)
print('Val data accuracy - ', result['val_acc']*100)
print('Val data loss - ', result['val_loss'])


# # Plots

# In[23]:


def plot_accuracies(history):
    """ Plot the history of accuracies"""
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-b')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');
    plt.show()
    

plot_accuracies(history)


# In[24]:


def plot_losses(history):
    """ Plot the losses in each epoch"""
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-b')
    plt.plot(val_losses, '-r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs');
    plt.show()

plot_losses(history)


# In[26]:


# #save the model
# torch.save(model.state_dict(), 'symbol_classification_256.pth')


# In[27]:


# torch.save(model, 'symbol_classification_256.pth')


# In[28]:


# with open('history.txt','w') as fp:
#     fp.write(str(history))


# In[ ]:



#Plotting the confusion matrix of the test  result
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.figure(figsize=(20,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[30]:




y_pred = []
y_true = []

val_dl = DeviceDataLoader(DataLoader(val_dataset, batch_size*2), device)
for images, labels in val_dl:
        output = model(images) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# constant for classes
classes = [str(i) for i in range(0,25)]

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cf_matrix, classes) 


# In[ ]:


y_pred = []
y_true = []
test_dl = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
for images, labels in test_dl:
        output = model(images) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# constant for classes
classes = [str(i) for i in range(0,25)]

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cf_matrix, classes) 

