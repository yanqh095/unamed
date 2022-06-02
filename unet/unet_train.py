import numpy as np
import mmcv
import os

from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch

os.chdir('/data1/qhong/seg/mmdetection/data')

def 

class CustomDataset(Dataset):
    def __init__(self, imgs, masks):
        self.imgs = imgs
        self.masks = masks
    
    def __getitem__(self, index):
        n = len(self.imgs)
        index = index % n
        img = self.imgs[index]
        img = img.reshape((1,)+img.shape)
        mask = self.masks[index]
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }
        
    def __len__(self):
        return len(self.imgs)

batch_size = 16
lr = 0.001
dataset = CustomDataset(crops, masks)
n_val = int(len(dataset) * 0.2)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
#train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
criterion = nn.CrossEntropyLoss()

def train(data_path, anno_path):
    # load annotation / training
    anno = mmcv.load(anno_path)

    
device = torch.device('cuda:1')
net.to(device=device)
epochs = 50
global_step = 0
for epoch in range(epochs):
    net.train()
    epoch_loss = 0
    for batch in train_loader:
        imgs = batch['image']
        true_masks = batch['mask']

        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)

        masks_pred = net(imgs)
        loss = criterion(masks_pred, true_masks)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

        global_step += 1
        if global_step % (n_train // (10 * batch_size)) == 0:
            print('step: %d, loss: %f'%(global_step, epoch_loss))
            val_score = eval_net(net, val_loader, device)
            scheduler.step(val_score)