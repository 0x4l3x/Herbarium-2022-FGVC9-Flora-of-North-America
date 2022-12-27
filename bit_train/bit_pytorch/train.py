# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Fine-tune a BiT model on some downstream dataset."""
#!/usr/bin/env python3
# coding: utf-8
from ast import arg
from fileinput import filename
from importlib.resources import path
from os.path import join as pjoin  # pylint: disable=g-importing-member
import time
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
from matplotlib.colors import LinearSegmentedColormap


import bit_pytorch.fewshot as fs
import bit_pytorch.lbtoolbox as lb
import bit_pytorch.models as models

import bit_common
import bit_hyperrule

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


def splitHerbarium(train_file):
  kf = StratifiedKFold(n_splits=5)
  for fold_, (train_idx, valid_idx) in enumerate(kf.split(X=train_file, y=train_file['category_id'])):
    print(f"{'='*40} Fold: {fold_+1} / {5} {'='*40}")
    train = train_file.loc[train_idx]
    valid = train_file.loc[valid_idx]
  
  #train, valid, ytrain, yvalid = train_test_split(train_file, train_file["category_id"], test_size=0.2, shuffle=True, random_state=42)
  print(len(train), len(valid))
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
        with open(pjoin(data_dir,"{}_metadata.json".format(split))) as f:
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
    precrop= 384
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
        labels = np.int64(self.catIds[idx])

        # Load image, resize and transform
        img = Image.open(pjoin(self.datadir, self.splitPath, self.file_paths[idx]))
        # img = img.resize((self.image_size,self.image_size))
        img = self.transforms(img)

        # Return image and class
        return img, labels

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
        return train_set, valid_set, train_loader, valid_loader

def topk(output, target, ks=(1,)):
  """Returns one boolean vector for each k, whether the target is within the output's top-k."""
  _, pred = output.topk(max(ks), 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))
  return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
  """Variant of itertools.cycle that does not save iterates."""
  while True:
    for i in iterable:
      yield i


def mktrainval(args, logger):
  """Returns train and validation datasets."""
  micro_batch_size = args.batch // args.batch_split
  precrop, crop = bit_hyperrule.get_resolution_from_dataset(args.dataset)
  train_tx = tv.transforms.Compose([
      tv.transforms.Resize((precrop, precrop)),
      tv.transforms.RandomCrop((crop, crop)),
      tv.transforms.RandomHorizontalFlip(),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])
  val_tx = tv.transforms.Compose([
      tv.transforms.Resize((crop, crop)),
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
  ])

  if args.dataset == "cifar10":
    train_set = tv.datasets.CIFAR10(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR10(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "cifar100":
    train_set = tv.datasets.CIFAR100(args.datadir, transform=train_tx, train=True, download=True)
    valid_set = tv.datasets.CIFAR100(args.datadir, transform=val_tx, train=False, download=True)
  elif args.dataset == "imagenet2012":
    train_set = tv.datasets.ImageFolder(pjoin(args.datadir, "train"), train_tx)
    valid_set = tv.datasets.ImageFolder(pjoin(args.datadir, "val"), val_tx)
  elif args.dataset == "herbarium":
    train_set, valid_set, train_loader, valid_loader,  = generateLoaders(micro_batch_size, args.datadir)
    return train_set, valid_set, train_loader, valid_loader

  else:
    raise ValueError(f"Sorry, we have not spent time implementing the "
                     f"{args.dataset} dataset in the PyTorch codebase. "
                     f"In principle, it should be easy to add :)")

  if args.examples_per_class is not None:
    logger.info(f"Looking for {args.examples_per_class} images per class...")
    indices = fs.find_fewshot_indices(train_set, args.examples_per_class)
    train_set = torch.utils.data.Subset(train_set, indices=indices)

  logger.info(f"Using a training set with {len(train_set)} images.")
  logger.info(f"Using a validation set with {len(valid_set)} images.")

  micro_batch_size = args.batch // args.batch_split

  valid_loader = torch.utils.data.DataLoader(
      valid_set, batch_size=micro_batch_size, shuffle=False,
      num_workers=args.workers, pin_memory=True, drop_last=False)

  if micro_batch_size <= len(train_set):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)
  else:
    # In the few-shot cases, the total dataset size might be smaller than the batch-size.
    # In these cases, the default sampler doesn't repeat, so we need to make it do that
    # if we want to match the behaviour from the paper.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=micro_batch_size, num_workers=args.workers, pin_memory=True,
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=micro_batch_size))

  return train_set, valid_set, train_loader, valid_loader

def run_eval(model, data_loader, device, chrono, logger, step):
  # switch to evaluate mode
  model.eval()

  logger.info("Running validation...")
  logger.flush()

  all_c, all_top1, all_top5 = [], [], []
  end = time.time()
  for b, (x, y) in enumerate(data_loader):
    with torch.no_grad():
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # measure data loading time
      chrono._done("eval load", time.time() - end)

      # compute output, measure accuracy and record loss.
      with chrono.measure("eval fprop"):
        logits = model(x)
        c = torch.nn.CrossEntropyLoss(reduction='none')(logits, y)
        top1, top5 = topk(logits, y, ks=(1, 5))
        all_c.extend(c.cpu())  # Also ensures a sync point.
        all_top1.extend(top1.cpu())
        all_top5.extend(top5.cpu())

    # measure elapsed time
    end = time.time()

  model.train()
  logger.info(f"Validation@{step} loss {np.mean(all_c):.5f}, "
              f"top1 {np.mean(all_top1):.2%}, "
              f"top5 {np.mean(all_top5):.2%}")
  logger.flush()
  return all_c, all_top1, all_top5
  
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

def mixup_data(x, y, l):
  """Returns mixed inputs, pairs of targets, and lambda"""
  indices = torch.randperm(x.shape[0]).to(x.device)

  mixed_x = l * x + (1 - l) * x[indices]
  y_a, y_b = y, y[indices]
  return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
  return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

def main(args):
  logger = bit_common.setup_logger(args)
  print("uWUWUWUWUWUUW")
  print(args.save)
  # Lets cuDNN benchmark conv implementations and choose the fastest.
  # Only good if sizes stay the same within the main loop!
  torch.backends.cudnn.benchmark = True

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  logger.info(f"Going to train on {device}")

  train_set, valid_set, train_loader, valid_loader = mktrainval(args, logger)

  logger.info(f"Loading model from {args.model}.npz")
  model = models.KNOWN_MODELS[args.model](head_size=15505, zero_head=True)
  model.load_from(np.load(f"{args.model}.npz"))

  logger.info("Moving model onto all GPUs")
  model = torch.nn.DataParallel(model)

  # Optionally resume from a checkpoint.
  # Load it to CPU first as we'll move the model to GPU later.
  # This way, we save a little bit of GPU memory when loading.
  step = 0

  # Note: no weight-decay!
  optim = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

  # Resume fine-tuning if we find a saved model.
  savename = pjoin(args.logdir, args.name, "ui.pth.tar")
  try:
    logger.info(f"Model will be saved in '{savename}'")
    checkpoint = torch.load(savename, map_location="cpu")
    logger.info(f"Found saved model to resume from at '{savename}'")

    step = checkpoint["step"]
    model.load_state_dict(checkpoint["model"])
    optim.load_state_dict(checkpoint["optim"])
    logger.info(f"Resumed at step {step}")
  except FileNotFoundError:
    logger.info("Fine-tuning from BiT")
  model = model.to(device)
  optim.zero_grad()
  model.train()
  mixup = bit_hyperrule.get_mixup(len(train_set))
  cri = torch.nn.CrossEntropyLoss().to(device)

  logger.info("Starting training!")
  chrono = lb.Chrono()
  accum_steps = 0
  mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
  end = time.time()
  logss=[]
  with lb.Uninterrupt() as u:
    for x, y in recycle(train_loader):
      # measure data loading time, which is spent in the `for` statement.
      chrono._done("load", time.time() - end)

      if u.interrupted:
        break

      # Schedule sending to GPU(s)
      x = x.to(device, non_blocking=True)
      y = y.to(device, non_blocking=True)

      # Update learning-rate, including stop training if over.
      lr = bit_hyperrule.get_lr(step, len(train_set), args.base_lr)
      if lr is None:
        break
      for param_group in optim.param_groups:
        param_group["lr"] = lr

      if mixup > 0.0:
        x, y_a, y_b = mixup_data(x, y, mixup_l)

      # compute output
      with chrono.measure("fprop"):
        logits = model(x)
        if mixup > 0.0:
          c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
        else:
          c = cri(logits, y)
        c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

      # Accumulate grads
      with chrono.measure("grads"):
        (c / args.batch_split).backward()
        accum_steps += 1

      accstep = f" ({accum_steps}/{args.batch_split})" if args.batch_split > 1 else ""
      logger.info(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")  # pylint: disable=logging-format-interpolation
      logger.flush()
      logss.append(f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})")

      # Update params
      if accum_steps == args.batch_split:
        with chrono.measure("update"):
          optim.step()
          optim.zero_grad()
        step += 1
        accum_steps = 0
        # Sample new mixup ratio for next batch
        mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

        # Run evaluation and save the model.
        if args.eval_every and step % args.eval_every == 0:
          run_eval(model, valid_loader, device, chrono, logger, step)
          if args.save:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim" : optim.state_dict(),
            }, savename)

      end = time.time()

    # Final eval at end of training.
    print(logss)
    run_eval(model, valid_loader, device, chrono, logger, step='end')
    
  torch.save({
                "step": step,
                "model": model.state_dict(),
                "optim" : optim.state_dict(),
  }, "/home/amorante/lastVersion.pth.tar")

  inferenceTestSet(model, args.batch // args.batch_split, args.datadir)

if __name__ == "__main__":
  parser = bit_common.argparser(models.KNOWN_MODELS.keys())
  parser.add_argument("--datadir", required=True,
                      help="Path to the ImageNet data folder, preprocessed for torchvision.")
  parser.add_argument("--workers", type=int, default=8,
                      help="Number of background threads used to load data.")
  parser.add_argument("--no-save", dest="save", action="store_false")
  parser.add_argument("--save", dest="save", action="store_true")

  main(parser.parse_args())
