#!/usr/bin/env python3
# coding: utf-8
from os.path import join as pjoin  # pylint: disable=g-importing-member
import json
from tqdm import tqdm
import csv
import pickle
import gc
import GPUtil
import numpy as np
import pandas as pd
import torch
import torchvision as tv
from PIL import Image
import torch.nn as nn
from datetime import datetime
from tensorboardX import SummaryWriter
from matplotlib.colors import LinearSegmentedColormap
import copy

from torch import optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR,ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt

import os
import glob
import numpy as np
from tqdm.auto import tqdm
from joblib import Parallel, delayed

#######          DATA METHODS           ###########

def splitHerbarium(train_file):
  kf = StratifiedKFold(n_splits=5)
  for fold_, (train_idx, valid_idx) in enumerate(kf.split(X=train_file, y=train_file['category_id'])):
    print(f"{'='*40} Fold: {fold_+1} / {5} {'='*40}")
    train = train_file.loc[train_idx]
    valid = train_file.loc[valid_idx]
  
  #train, valid, ytrain, yvalid = train_test_split(train_file, train_file["category_id"], test_size=0.2, shuffle=True, random_state=42)
  return train, valid

def generateDataSets(datadir,split):
    if split == "train":
        with open(pjoin(datadir,"train_metadata.json"), "r", encoding="ISO-8859-1") as file:
            train = json.load(file)
        train_img = pd.DataFrame(train['images'])
        train_ann = pd.DataFrame(train['annotations'])
        train_file = train_img.merge(train_ann, on='image_id')

        train, valid= splitHerbarium(train_file)
        train_set=HerbariumData(datadir, train["file_name"], train["category_id"], "train")
        valid_set=HerbariumData(datadir, valid["file_name"], valid["category_id"], "valid")
        return train_set, valid_set
        #return np.array(train_df["file_name"]), np.array(train_df["category_id"])
    else:
        with open(pjoin(datadir,"{}_metadata.json".format(split))) as f:
          file_data = json.load(f)
        test= pd.json_normalize(file_data)
        test_set= HerbariumData(datadir, test["file_name"], test["image_id"], "test")
        return test_set
        #return np.array(test_df["file_name"]), np.array(test_df["image_id"])

def _color_means(img_path):
    img = plt.imread(img_path)
    means = {i: np.mean(img[..., i]) / 255.0 for i in range(3)}
    std = {i: np.std(img[..., i]) / 255.0 for i in range(3)}
    return means, std

def get_transforms(split):#
    precrop= 380
    crop= 350
    img_color_mean, img_color_std= [0.7783196839606584, 0.7565902223742913, 0.7097361636328653], [0.246667634451421, 0.25063209592622404, 0.2535765914495854]
    if split=="train":
      return tv.transforms.Compose([
          tv.transforms.Resize((precrop, precrop)),
          tv.transforms.CenterCrop((crop,crop)),
          tv.transforms.RandomHorizontalFlip(p=0.5),
          tv.transforms.ToTensor(),
          tv.transforms.Normalize(img_color_mean,img_color_std)
          ])
    #same transformations for test and validation
    else:
      return tv.transforms.Compose([
        tv.transforms.Resize((precrop, precrop)),
        tv.transforms.CenterCrop((crop,crop)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(img_color_mean,img_color_std)
        ])

class HerbariumData(torch.utils.data.Dataset):
    """
    Custom dataset class for this competition's data
    """
    def __init__(self, datadir, file_pths, class_list, split):
        self.datadir=datadir
        self.file_paths = np.array(file_pths)
        if split=="valid":
            self.splitPath="train_images"
        else:
            self.splitPath=split+"_images"
        self.catIds=  np.array(class_list)
        self.classes = 15505
        self.transforms = get_transforms(split)
        self.total_img_count = self.file_paths.shape[0]

    def __len__(self):
        return self.total_img_count

    def __getitem__(self,idx):
        # Get id or classindex
        class_ = np.int64(self.catIds[idx])

        # Load image, resize and transform
        img = Image.open(pjoin(self.datadir, self.splitPath, self.file_paths[idx]))
        # img = img.resize((self.image_size,self.image_size))
        img = self.transforms(img)

        # Return image and class
        return img, class_

    def getFileAndClass(self):
        return self.file_paths, self.catIds

# ENABLING PINNED MEMORY FOR FASTER LOADING
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

def collate_fn(samples):
    pixel_values = torch.stack([sample[0] for sample in samples])
    labels = torch.tensor([sample[1] for sample in samples])

    batch = {"pixel_values": pixel_values, "labels": labels}
    batch = {k: v.numpy() for k, v in batch.items()}

    return batch
########
def generateLoaders(micro_batch_size,  datadir, split="train"):
    if split == "test":
        test_set=generateDataSets(datadir, "test")
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=fixedBatchSize, shuffle=False,
            num_workers=8, pin_memory=True
        )
        return test_loader
    else :
        train_set , valid_set= generateDataSets(datadir, "train")
        train_loader= torch.utils.data.DataLoader(train_set ,batch_size=micro_batch_size, shuffle=True, 
            pin_memory=True, num_workers=8
        )
        valid_loader= torch.utils.data.DataLoader(valid_set ,batch_size=micro_batch_size, shuffle=True, 
            pin_memory=True, num_workers=8
        )
        return train_loader, valid_loader, train_set.classes

######## STATS #########
def calculateMeanStd():
    PATH_DATASET="/home/amorante/herbarium"
    images = glob.glob(pjoin(PATH_DATASET, "train_images", "*", "*", "*.jpg"))
    #control ammount of images using images[:ammount]
    clr_mean_std = Parallel(n_jobs=os.cpu_count())(delayed(_color_means)(fn) for fn in tqdm(images))

    img_color_mean = pd.DataFrame([c[0] for c in clr_mean_std])
    img_color_std = pd.DataFrame([c[1] for c in clr_mean_std])
    print(img_color_mean.describe())
    print(img_color_std.describe())

    img_color_mean = list(img_color_mean.T["mean"])
    img_color_std = list(img_color_std.T["mean"])
    print(img_color_mean, img_color_std)

####### TRAIN ############

def train(model, fixedBatchSize, loaders, criterion):
    loss_evolution= []
    steps =0
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  
        else:
            model.eval()   
        correct_class, total_loss, total_num  = 0, 0, 0 
        pbar = tqdm(enumerate(loaders[phase]), total=len(loaders[phase]), desc='{} epoch {}'.format(phase, epoch))
        for batch_idx, (data, class_) in pbar:
            data  = data.to(device)
            class_= class_.to(device)
            with torch.set_grad_enabled(phase == 'train'):
                class_output  = model(data)
                loss_val = criterion(class_output, class_) / fixedBatchSize
                if phase == 'train':
                    loss_val.backward() 
                    if (batch_idx+1) % int(fixedBatchSize) == 0:
                        optimizer.step()
                        steps += 1
                        loss_evolution.append(loss_val.item())
                        optimizer.zero_grad()      
                _, predicted = torch.max(class_output.data, 1)
            correct_class += (predicted == class_).sum().item()
            total_loss += loss_val.item()
            total_num += data.shape[0]
            pbar.set_postfix(avg_acc=float(correct_class)/total_num)
        
        class_acc = float(correct_class)/total_num
        avg_loss = total_loss/(batch_idx+ 1) # Average per batch
        if phase == "train": 
            train_dict= {"train_epoch_loss": avg_loss, "epoch_acc":class_acc}
        else :
            val_dict= {"val_epoch_loss": avg_loss, "epoch_acc":class_acc}
    print(train_dict,"\n", val_dict, "\n")
    return train_dict, val_dict, steps, loss_evolution
###### MODEL ########

def buildModel(num_classes):
    pretrained_model = tv.models.resnet18(pretrained=True)
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, num_classes)
    return pretrained_model

######### INFERENCE #########
def inferenceTestSet(model, fixedBatchSize, datadir):
    model.eval()
    output_file = open("/home/amorante/submission.csv", 'w')
    csv_writer = csv.writer(output_file, delimiter=',')
    csv_writer.writerow(["Id","Predicted"])
    test_loader= generateLoaders(fixedBatchSize, datadir, "test")
    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Inference')
    with torch.no_grad():
        for batch_idx, (x, id_lab) in pbar:
            x = x.to(device)
            logits = model(x)
            _, predicted = torch.max(logits, 1)
            preds=predicted.cpu().numpy()
            id_lab=id_lab.numpy()
            for i in range(preds.shape[0]):
                csv_writer.writerow(['{:d}'.format(int(id_lab[i])),'{:d}'.format(int(preds[i]))])
    output_file.close()

# 0. Calculate stats.

#calculateMeanStd()

# 1. Seed all processes
RANDOM_SEED = 9999
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# 2. Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))
if device == "cuda":
    torch.backends.cudnn.benchmark = True
# 3. Setup logging
dt_string = datetime.now().strftime("%d%m%Y%H%M%S")
dir_name = "baseline_run/run_id_{}".format(dt_string)
logger = SummaryWriter(log_dir=dir_name)

# 4. Load data
batch_size = 8
fixedBatchSize = (512//batch_size)
print(fixedBatchSize)
datadir = "/home/amorante/herbarium" # PATH TO DATA 

# train_data_loader, num_classes, qes = get_loaders(data_dir,batch_size=batch_size)
train_loader, valid_loader, num_classes = generateLoaders(fixedBatchSize, datadir)


# 5. Load model, using pretrained imagenet weights, and put on device
model = buildModel(num_classes)
model.to(device)

# 6.  Setup optimization, learning rate scheduler, and loss function
num_epochs = 20
optimizer = optim.SGD(model.parameters(), lr=0.5,momentum=.9,weight_decay=0.0001,nesterov=True)
scheduler = ReduceLROnPlateau(optimizer, patience=2)
criterion = nn.CrossEntropyLoss()


# 7. Start train/test loop
best_acc, steps = 0, 0
loss_evolution=[]
logging_dict = {}
logs=[]
loaders={'train':train_loader , 'val':valid_loader}
for epoch in range(num_epochs):
    # Train one epoch, get loss and acc
    torch.cuda.empty_cache()
    train_dict, valid_dict, s, lev = train(model,fixedBatchSize, loaders, criterion)
    scheduler.step(valid_dict["val_epoch_loss"])
    #stats
    steps += s
    loss_evolution.append(lev)
    torch.cuda.empty_cache()
    logging_dict["{}_e={}".format("train",epoch)] = train_dict
    logging_dict["{}_e={}".format("valid",epoch)] = valid_dict
    logs.append(train_dict)
    logs.append(valid_dict)
    # Save model every epoch (could ensemble later)
    model_name = "r50epoch_{}_ca_{:.3f}.pth".format(epoch,valid_dict["epoch_acc"])
    print(model_name)
    if valid_dict["epoch_acc"] > best_acc:
        best_model_pth = pjoin(datadir,model_name)+".tar"
        best_acc = valid_dict["epoch_acc"]
        torch.save({
                        "step": s,
                        "model": model.state_dict(),
                        "optim" : optimizer.state_dict(),
        }, best_model_pth)
##########
inferenceTestSet(model, fixedBatchSize, datadir)
with open('lossEvol.pkl', 'wb') as f:
    pickle.dump(loss_evolution, f)
with open('logs.pkl', 'wb') as f:
    pickle.dump(logs, f)
print(steps)
print(loss_evolution)
print("\n\n",logs,"\n\n")
##########                    
