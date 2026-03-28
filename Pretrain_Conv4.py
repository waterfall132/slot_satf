#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretrain_Conv4.py
=================
Conv4/Conv64F backbone pretraining script for CUB / miniImageNet style datasets.

Notes:
- Reuses the existing dataset logic from Pretrain_ResNet12.py
- Saves both full model checkpoint and encoder-only checkpoint
- encoder_model supports: Conv4, Conv64F, Conv64F_Local
- For Conv4, we map to backbone.Conv64F()
"""

from __future__ import print_function
import argparse, os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import ImageFile, Image as PILImage
import csv as csv_module, sys

sys.dont_write_bytecode = True
import models.backbone as backbone

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# ─────────────────────────── 参数 ────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Conv4 Pretrain')
parser.add_argument('--dataset_dir',  default='/root/autodl-tmp/CUB200/CUB_200_2011_FewShot')
parser.add_argument('--data_name',    default='CubBird',
                    help='miniImageNet | CubBird | StanfordDog | StanfordCar | tieredImageNet')
parser.add_argument('--imageSize',    type=int, default=84)
parser.add_argument('--workers',      type=int, default=8)

# Conv4 推荐配置
parser.add_argument('--epochs',          type=int,   default=300)
parser.add_argument('--batch_size',      type=int,   default=32)
parser.add_argument('--lr',              type=float, default=0.001)
parser.add_argument('--lr_decay_epochs', nargs='+', type=int, default=[100, 200])
parser.add_argument('--lr_decay_rate',   type=float, default=0.1)
parser.add_argument('--weight_decay',    type=float, default=5e-4)
parser.add_argument('--momentum',        type=float, default=0.9)
parser.add_argument('--nesterov',        action='store_true', default=True)
parser.add_argument('--cosine',          action='store_true', default=False)
parser.add_argument('--encoder_model',   default='Conv4', choices=['Conv4', 'Conv64F', 'Conv64F_Local'])
parser.add_argument('--num_classes',     type=int, default=100)
parser.add_argument('--outf',            default='./pretrain_results_conv4')
parser.add_argument('--print_freq',      type=int, default=50)
parser.add_argument('--cuda',            action='store_true', default=True)

opt = parser.parse_args()
cudnn.benchmark = True

NUM_CLASSES = {'miniImageNet':64,'tieredImageNet':351,'CubBird':100,'StanfordDog':70,'StanfordCar':130}
if opt.data_name in NUM_CLASSES:
    opt.num_classes = NUM_CLASSES[opt.data_name]
    print(f'[INFO] {opt.data_name}: num_classes = {opt.num_classes}')

opt.outf = os.path.join(opt.outf, opt.data_name)
os.makedirs(opt.outf, exist_ok=True)

# ─────────────────────────── 工具函数 ────────────────────────────────────────
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.val=self.avg=self.sum=self.count=0
    def update(self,val,n=1):
        self.val=val; self.sum+=val*n; self.count+=n; self.avg=self.sum/self.count

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk=max(topk); bs=target.size(0)
        _,pred=output.topk(maxk,1,True,True); pred=pred.t()
        correct=pred.eq(target.view(1,-1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum(0).mul_(100.0/bs) for k in topk]

def save_checkpoint(state, is_best, outf, filename='checkpoint.pth.tar'):
    path=os.path.join(outf,filename); torch.save(state,path)
    if is_best:
        import shutil; shutil.copyfile(path,os.path.join(outf,'pretrain_best.pth.tar'))

# ─────────────────────────── 数据增广 ────────────────────────────────────────
def get_transforms(split='train', imageSize=84):
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    if split=='train':
        return transforms.Compose([
            transforms.RandomResizedCrop(imageSize),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4,0.4,0.4,0.1),
            transforms.ToTensor(), transforms.Normalize(mean,std)])
    else:
        return transforms.Compose([
            transforms.Resize(int(imageSize*1.15)),
            transforms.CenterCrop(imageSize),
            transforms.ToTensor(), transforms.Normalize(mean,std)])

# ─────────────────────────── 数据集 ──────────────────────────────────────────
class CSVDataset(Dataset):
    def __init__(self, dataset_dir, csv_file, transform=None):
        self.dataset_dir=dataset_dir; self.transform=transform; self.samples=[]
        class_img_dict={}
        with open(csv_file) as f:
            reader=csv_module.reader(f); next(reader)
            for row in reader:
                class_img_dict.setdefault(row[1],[]).append(row[0])
        class_list=sorted(class_img_dict.keys())
        self.class_to_idx={c:i for i,c in enumerate(class_list)}
        self.classes=class_list
        for cls,imgs in class_img_dict.items():
            idx=self.class_to_idx[cls]
            for img_name in imgs:
                self.samples.append((os.path.join(dataset_dir,'images',img_name),idx))
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        img_path,label=self.samples[index]
        img=PILImage.open(img_path).convert('RGB')
        if self.transform: img=self.transform(img)
        return img,label

def get_dataloader(opt):
    train_csv=os.path.join(opt.dataset_dir,'train.csv')
    if os.path.isfile(train_csv):
        print(f'[INFO] CSV dataset: {train_csv}')
        full=CSVDataset(opt.dataset_dir,train_csv)
        from collections import defaultdict; import random as rnd
        cls_to_idx=defaultdict(list)
        for idx,(_,label) in enumerate(full.samples): cls_to_idx[label].append(idx)
        train_idx,val_idx=[],[]
        rnd.seed(42)
        for label,indices in cls_to_idx.items():
            rnd.shuffle(indices); split=max(1,int(len(indices)*0.1))
            val_idx.extend(indices[:split]); train_idx.extend(indices[split:])

        class TransformSubset(Dataset):
            def __init__(self,base,indices,transform):
                self.base=base; self.indices=indices; self.transform=transform
            def __len__(self): return len(self.indices)
            def __getitem__(self,i):
                img_path,label=self.base.samples[self.indices[i]]
                img=PILImage.open(img_path).convert('RGB')
                if self.transform: img=self.transform(img)
                return img,label

        train_ds=TransformSubset(full,train_idx,get_transforms('train',opt.imageSize))
        val_ds  =TransformSubset(full,val_idx,  get_transforms('val',  opt.imageSize))
        actual_classes=len(full.classes)
    else:
        print('[INFO] ImageFolder dataset.')
        train_ds=datasets.ImageFolder(os.path.join(opt.dataset_dir,'train'),get_transforms('train',opt.imageSize))
        val_ds  =datasets.ImageFolder(os.path.join(opt.dataset_dir,'val'),  get_transforms('val',  opt.imageSize))
        actual_classes=len(train_ds.classes)

    if actual_classes!=opt.num_classes:
        print(f'[WARNING] Found {actual_classes} classes, overriding.')
        opt.num_classes=actual_classes

    train_loader=DataLoader(train_ds,batch_size=opt.batch_size,shuffle=True,
                            num_workers=opt.workers,pin_memory=True,drop_last=True)
    val_loader  =DataLoader(val_ds,  batch_size=opt.batch_size,shuffle=False,
                            num_workers=opt.workers,pin_memory=True,drop_last=False)
    print(f'[INFO] classes={actual_classes}, train={len(train_ds)}, val={len(val_ds)}')
    return train_loader, val_loader

# ─────────────────────────── 模型 ────────────────────────────────────────────
class ConvPretrainNet(nn.Module):
    def __init__(self, encoder_model='Conv4', num_classes=100):
        super().__init__()
        if encoder_model in ['Conv4', 'Conv64F']:
            self.encoder = backbone.Conv64F()
            feat_dim = 64
        elif encoder_model == 'Conv64F_Local':
            self.encoder = backbone.Conv64F_Local()
            feat_dim = 64
        else:
            raise ValueError(f'Unsupported encoder_model: {encoder_model}')

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def build_model(opt):
    return ConvPretrainNet(encoder_model=opt.encoder_model, num_classes=opt.num_classes)

# ─────────────────────────── 训练/验证 ───────────────────────────────────────
def train_one_epoch(loader,model,criterion,optimizer,epoch,opt,F_txt):
    losses=AverageMeter(); top1=AverageMeter(); top5=AverageMeter()
    model.train()
    for i,(images,targets) in enumerate(loader):
        if opt.cuda:
            images=images.cuda(non_blocking=True)
            targets=targets.cuda(non_blocking=True)
        output=model(images)
        loss=criterion(output,targets)
        prec1,prec5=accuracy(output,targets,topk=(1,5))
        losses.update(loss.item(),images.size(0))
        top1.update(prec1.item(),images.size(0))
        top5.update(prec5.item(),images.size(0))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if i%opt.print_freq==0:
            msg=f'Epoch[{epoch}][{i}/{len(loader)}] Loss {losses.avg:.4f}  Acc@1 {top1.avg:.2f}  Acc@5 {top5.avg:.2f}'
            print(msg)
            if F_txt: print(msg,file=F_txt)
    return top1.avg, losses.avg

def validate(loader,model,criterion,epoch,opt,F_txt):
    losses=AverageMeter(); top1=AverageMeter(); top5=AverageMeter()
    model.eval()
    with torch.no_grad():
        for images,targets in loader:
            if opt.cuda:
                images=images.cuda(non_blocking=True)
                targets=targets.cuda(non_blocking=True)
            output=model(images)
            loss=criterion(output,targets)
            prec1,prec5=accuracy(output,targets,topk=(1,5))
            losses.update(loss.item(),images.size(0))
            top1.update(prec1.item(),images.size(0))
            top5.update(prec5.item(),images.size(0))
    msg=f'Val[{epoch}] Loss {losses.avg:.4f}  Acc@1 {top1.avg:.2f}  Acc@5 {top5.avg:.2f}'
    print(msg)
    if F_txt: print(msg,file=F_txt)
    return top1.avg, losses.avg

# ─────────────────────────── 主函数 ──────────────────────────────────────────
if __name__=='__main__':
    F_txt=open(os.path.join(opt.outf,'pretrain_log.txt'),'a+')
    print(opt); print(opt,file=F_txt)

    train_loader,val_loader=get_dataloader(opt)
    model=build_model(opt)
    if opt.cuda:
        model=model.cuda()
    criterion=nn.CrossEntropyLoss().cuda() if opt.cuda else nn.CrossEntropyLoss()

    optimizer=optim.SGD(model.parameters(),lr=opt.lr,momentum=opt.momentum,
                        weight_decay=opt.weight_decay,nesterov=opt.nesterov)

    if opt.cosine:
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,opt.epochs,eta_min=opt.lr*(opt.lr_decay_rate**3))
        print('[INFO] CosineAnnealingLR')
    else:
        scheduler=torch.optim.lr_scheduler.MultiStepLR(
            optimizer,milestones=opt.lr_decay_epochs,gamma=opt.lr_decay_rate)
        print(f'[INFO] MultiStepLR milestones={opt.lr_decay_epochs}')

    best_acc=0.0
    print(f'\n=== Pretrain {opt.encoder_model} on {opt.data_name} | lr={opt.lr} | bs={opt.batch_size} | epochs={opt.epochs} ===')

    for epoch in range(opt.epochs):
        cur_lr=optimizer.param_groups[0]['lr']
        print(f'\nEpoch {epoch}/{opt.epochs}  lr={cur_lr:.6f}')

        train_acc,_=train_one_epoch(train_loader,model,criterion,optimizer,epoch,opt,F_txt)
        val_acc,_  =validate(val_loader,model,criterion,epoch,opt,F_txt)
        scheduler.step()

        is_best=val_acc>best_acc
        best_acc=max(val_acc,best_acc)
        state={
            'epoch':epoch,
            'encoder_model':opt.encoder_model,
            'num_classes':opt.num_classes,
            'model':model.state_dict(),
            'best_acc':best_acc,
            'optimizer':optimizer.state_dict()
        }

        save_checkpoint(state,is_best,opt.outf,filename='pretrain_last.pth.tar')
        if epoch>0 and epoch%50==0:
            save_checkpoint(state,False,opt.outf,filename=f'pretrain_epoch_{epoch}.pth.tar')

        msg=f'[Epoch {epoch}] train={train_acc:.2f}  val={val_acc:.2f}  best={best_acc:.2f}'
        print(msg); print(msg,file=F_txt)

    print(f'\n=== Done. Best Val Acc: {best_acc:.2f} ===')

    encoder_state = {}
    for k,v in model.state_dict().items():
        if k.startswith('encoder.'):
            encoder_state[k[len('encoder.'):]] = v

    encoder_path=os.path.join(opt.outf,'encoder_pretrained.pth.tar')
    torch.save({'encoder_model':opt.encoder_model,'encoder_state_dict':encoder_state,
                'best_acc':best_acc},encoder_path)
    print(f'[INFO] Encoder saved → {encoder_path}')
    F_txt.close()
