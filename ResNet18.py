#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' nvidia-smi')


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'


# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('bmh')
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from torchvision import models
import albumentations as A
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
from PIL import Image
import json
import timm
from torchvision import models as tvmodels
import seaborn as sns


# In[4]:


def get_device():
    ''' Get device (if GPU is available, use GPU) '''
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# In[5]:


main_dir = "/scratch/uceegs0/dataset/"
with open(os.path.join(main_dir, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    
print(json.dumps(map_classes, indent=4))


# In[6]:


input_dir = os.listdir(os.path.join(main_dir, "train_images"))
print(f"Number of train images: {len(input_dir)}")
input_dir


# 
# <font color=black size=5 face=雅黑>**Datasplit**</font>

# In[7]:


train_dir = os.path.join(main_dir, "train_images")
train_df = pd.read_csv(os.path.join(main_dir, "train.csv"))
train_df


# In[8]:


proportion = train_df['label'].value_counts(normalize=False).head(10)
proportion


# In[9]:


data = train_df['label'].astype(str).map(map_classes)
p = plt.hist(data)
plt.xticks(rotation='vertical')

plt.show()
plt.savefig("./datacount.png")


# In[10]:


ids = train_df['image_id'].values
labels = train_df['label'].values

X_train_id, X_rem_id, y_train, y_rem = train_test_split(ids, labels, train_size=0.6, random_state=0, stratify=labels)
X_val_id, X_test_id, y_val, y_test = train_test_split(X_rem_id, y_rem, test_size=0.5, random_state=0, stratify=y_rem)


# In[11]:


print('Number of training samples:', len(X_train_id))
print('Number of validation samples:', len(X_val_id))
print('Number of validation samples:', len(X_test_id))


# In[12]:


_, counts_train = np.unique(y_val, return_counts=True)
print('Training data distribution')
for i in range(5):
  print("{}: {:.3f}".format(i, counts_train[i]/len(y_val)))


# <font color=black size=5 face=雅黑>**Transforms**</font>

# In[13]:


# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    
])


train_tfm = transforms.Compose([
    transforms.RandomResizedCrop((128, 128), scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomRotation(180),
    transforms.RandomAffine(30),
    transforms.RandomPosterize(bits=2),
    transforms.RandomSolarize(threshold=192.0, p=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
])


# <font color=black size=5 face=雅黑>**Define Data class**</font>

# In[14]:


class CassavaDataset(Dataset):
    def __init__(self, data_dir, ids, labels, transform=None):
        self.data_dir = data_dir
        self.ids = ids
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.data_dir, self.ids[idx]))
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]    
        return (image, label)


# <font color=black size=5 face=雅黑>**Model**</font>

# In[15]:


class Residual_Block(nn.Module):
    def __init__(self, ic, oc, stride=1):
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(ic, oc, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(oc),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(oc, oc, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(oc),
        )
        
        self.relu = nn.ReLU(inplace=True)
    
        self.downsample = None
        if stride != 1 or (ic != oc):
            self.downsample = nn.Sequential(
                nn.Conv2d(ic, oc, kernel_size=1, stride=stride),
                nn.BatchNorm2d(oc),
            )
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.downsample:
            residual = self.downsample(x)
            
        out += residual
        return self.relu(out)
        
class Classifier(nn.Module):
    def __init__(self, block, num_layers, num_classes=11):
        super().__init__()
        self.preconv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.layer0 = self.make_residual(block, 32, 64,  num_layers[0], stride=2)
        self.layer1 = self.make_residual(block, 64, 128, num_layers[1], stride=2)
        self.layer2 = self.make_residual(block, 128, 256, num_layers[2], stride=2)
        self.layer3 = self.make_residual(block, 256, 512, num_layers[3], stride=2)
        
        #self.avgpool = nn.AvgPool2d(2)
        
        self.fc = nn.Sequential(            
            nn.Dropout(0.4),
            nn.Linear(512*4*4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 11),
        )
        
        
    def make_residual(self, block, ic, oc, num_layer, stride=1):
        layers = []
        layers.append(block(ic, oc, stride))
        for i in range(1, num_layer):
            layers.append(block(oc, oc))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # [3, 128, 128]
        out = self.preconv(x)  # [32, 64, 64]
        out = self.layer0(out) # [64, 32, 32]
        out = self.layer1(out) # [128, 16, 16]
        out = self.layer2(out) # [256, 8, 8]
        out = self.layer3(out) # [512, 4, 4]
        #out = self.avgpool(out) # [512, 2, 2]
        out = self.fc(out.view(out.size(0), -1)) 
        return out


# In[16]:


import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        
    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, dim=1)
        
        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        
        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]
        probs = (P*class_mask).sum(1).view(-1, 1)
        
        log_p = probs.log()
        
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
            
        return loss
    
class MyCrossEntropy(nn.Module):
    def __init__(self, class_num):
        pass


# In[17]:


myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


# In[18]:


batch_size = 128
num_layers = [2, 3, 3, 1]
alpha = torch.Tensor([1, 2.3, 0.66, 1, 1.1, 0.75, 2.3, 3.5, 1.1, 0.66, 1.4])
# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_dataset = CassavaDataset(train_dir, X_train_id, y_train, transform=train_tfm)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_dataset = CassavaDataset(train_dir, X_val_id, y_val, transform=test_tfm)
valid_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)


# <font color=black size=5 face=雅黑>**Train**</font>

# In[19]:


_exp_name = "ResNet18"


# In[42]:


get_ipython().run_cell_magic('time', '', 'train_loss_history = [] \nval_loss_history = []\ntrain_acc_history = []\nval_acc_history = []\nactual_epoch = 0\n\n# "cuda" only when GPUs are available.\ndevice = get_device() \n\n# The number of training epochs and patience.\nn_epochs = 1000\npatience = 48 # If no improvement in \'patience\' epochs, early stop\n\n# Initialize a model, and put it on the device specified.\nmodel = Classifier(Residual_Block, num_layers).to(device)\n\n# For the classification task, we use cross-entropy as the measurement of performance.\ncriterion = FocalLoss(11, alpha=alpha)\n\n# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.\noptimizer = torch.optim.Adam(model.parameters(), lr=0.03 weight_decay=1e-5) \nscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=16, T_mult=1)\n# Initialize trackers, these are not parameters and should not be changed\nstale = 0\nbest_acc = 0\n\nfor epoch in range(n_epochs):\n\n    # ---------- Training ----------\n    # Make sure the model is in train mode before training.\n    model.train()\n\n    # These are used to record information in training.\n    train_loss = []\n    train_accs = []\n\n    for batch in tqdm(train_loader):\n\n        # A batch consists of image data and corresponding labels.\n        imgs, labels = batch\n        #imgs = imgs.half()\n        #print(imgs.shape,labels.shape)\n\n        # Forward the data. (Make sure data and model are on the same device.)\n        logits = model(imgs.to(device))\n\n        # Calculate the cross-entropy loss.\n        # We don\'t need to apply softmax before computing cross-entropy as it is done automatically.\n        loss = criterion(logits, labels.to(device))\n\n        # Gradients stored in the parameters in the previous step should be cleared out first.\n        optimizer.zero_grad()\n        \n        # Compute the gradients for parameters.\n        loss.backward()\n\n        # Clip the gradient norms for stable training.\n        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)\n\n        # Update the parameters with computed gradients.\n        optimizer.step()\n\n        # Compute the accuracy for current batch.\n        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()\n\n        # Record the loss and accuracy.\n        train_loss.append(loss.item())\n        train_accs.append(acc)\n        \n    train_loss = sum(train_loss) / len(train_loss)\n    train_acc = sum(train_accs) / len(train_accs)\n    \n    train_loss_history.append(train_loss)\n    train_acc_history.append(train_acc)\n    \n    # Print the information.\n    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")\n    \n    scheduler.step()\n    \n    # ---------- Validation ----------\n    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.\n    model.eval()\n\n    # These are used to record information in validation.\n    valid_loss = []\n    valid_accs = []\n\n    # Iterate the validation set by batches.\n    for batch in tqdm(valid_loader):\n\n        # A batch consists of image data and corresponding labels.\n        imgs, labels = batch\n        #imgs = imgs.half()\n\n        # We don\'t need gradient in validation.\n        # Using torch.no_grad() accelerates the forward process.\n        with torch.no_grad():\n            logits = model(imgs.to(device))\n\n        # We can still compute the loss (but not the gradient).\n        loss = criterion(logits, labels.to(device))\n\n        # Compute the accuracy for current batch.\n        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()\n\n        # Record the loss and accuracy.\n        valid_loss.append(loss.item())\n        valid_accs.append(acc)\n        #break\n    \n    # The average loss and accuracy for entire validation set is the average of the recorded values.\n    valid_loss = sum(valid_loss) / len(valid_loss)\n    valid_acc = sum(valid_accs) / len(valid_accs)\n    \n    val_loss_history.append(valid_loss)\n    val_acc_history.append(valid_acc)\n    \n    # Print the information.\n    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")\n    actual_epoch = actual_epoch + 1\n\n    # update logs\n    if valid_acc > best_acc:\n        with open(f"./{_exp_name}_log.txt","a"):\n            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")\n    else:\n        with open(f"./{_exp_name}_log.txt","a"):\n            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")\n\n\n    # save models\n    if valid_acc > best_acc:\n        print(f"Best model found at epoch {epoch}, saving model")\n        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error\n        best_acc = valid_acc\n        stale = 0\n    else:\n        stale += 1\n        if stale > patience:\n            print(f"No improvment {patience} consecutive epochs, early stopping")\n            break\n            \nprint(\'Finished Training\')\n')


# In[43]:


epochs = list(range(1,actual_epoch+1))
plt.plot(epochs, train_loss_history, label='Avg Train Loss')
plt.plot(epochs, val_loss_history, label='Avg Val loss')
plt.title("Model Loss")
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
# plt.gca().set_ylim(0,1)
plt.legend();
plt.savefig(os.path.join("./Result/",_exp_name+"_loss.png"))


# In[44]:


def tensors_to_numpy(tensors):
    # Move each tensor from CUDA device to CPU and convert to a NumPy array
    numpy_arrays = [t.cpu().numpy() for t in tensors]

    # Convert the list of NumPy arrays to a NumPy array with shape (n, m)
    numpy_array = np.array(numpy_arrays)

    return numpy_array


# In[45]:


train_acc_history = tensors_to_numpy(train_acc_history)
val_acc_history = tensors_to_numpy(val_acc_history)


# In[46]:


plt.plot(epochs, train_acc_history, label='Train Acc')
plt.plot(epochs, val_acc_history, label='Val Acc')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
# plt.gca().set_ylim(0,1)
plt.legend();
plt.savefig(os.path.join("./Result/",_exp_name+"_acc.png"))
print('Best val acc is {:.3f} '.format(best_acc))


# <font color=black size=5 face=雅黑>**Test**</font>

# In[20]:


test_dataset = CassavaDataset(train_dir, X_test_id, y_test, transform=test_tfm)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)


# In[22]:


device = get_device() 
model_best = Classifier(Residual_Block, num_layers).to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in test_loader:
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()


# In[23]:


def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(test_dataset)+1)]
df["Category"] = prediction


# In[24]:


from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
print(accuracy_score(y_test, prediction))
print(precision_score(y_test, prediction, average='micro'))
print(recall_score(y_test, prediction, average='micro'))
print(f1_score(y_test, prediction, average='micro'))


# In[25]:


from sklearn.metrics import confusion_matrix
confusion_mtx = confusion_matrix(y_test, prediction) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
plt.savefig(os.path.join("./Result/",_exp_name+"_cm.png"))


# In[ ]:





# In[ ]:




