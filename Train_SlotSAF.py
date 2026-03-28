#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train_SlotSAF.py - SlotSAFNet v2 training script
"""

from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import time
from PIL import ImageFile
import sys

sys.dont_write_bytecode = True

import dataset.general_dataloader as FewShotDataloader
from models.slot_saf_net import SlotSAFNet
import utils

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/root/autodl-tmp/miniImageNet--ravi')
parser.add_argument('--data_name', default='miniImageNet')
parser.add_argument('--mode', default='train')
parser.add_argument('--outf', default='./results_SlotSAF')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--pretrained_encoder', default='', type=str)
parser.add_argument('--encoder_model', default='ResNet12')
parser.add_argument('--classifier_model', default='SlotSAFNet')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--train_aug', action='store_true', default=True)
parser.add_argument('--test_aug', action='store_true', default=False)
parser.add_argument('--episodeSize', type=int, default=1)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--episode_train_num', type=int, default=4000)
parser.add_argument('--episode_val_num', type=int, default=1000)
parser.add_argument('--episode_test_num', type=int, default=1000)
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)
parser.add_argument('--aug_shot_num', type=int, default=20)
parser.add_argument('--neighbor_k', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--cosine', type=bool, default=True)
parser.add_argument('--lr_decay_epochs', type=list, default=[60, 80])
parser.add_argument('--lr_decay_rate', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--adam', action='store_true', default=True)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--ngpu', type=int, default=1)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--num_slots', type=int, default=5)
parser.add_argument('--num_iters', type=int, default=3)
parser.add_argument('--beta', type=float, default=10.0)
parser.add_argument('--eta', type=float, default=0.1)
parser.add_argument('--lambda_bg', type=float, default=0.05)
parser.add_argument('--lambda_div', type=float, default=0.01)
parser.add_argument('--lambda_mask', type=float, default=0.001)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--query_seed_mode', type=str, default='self', choices=['self', 'global'])
parser.add_argument('--use_bg_logits', action='store_true', default=False)

opt = parser.parse_args()
opt.cuda = True
cudnn.benchmark = True


def compute_v2_losses(aux, labels, margin=0.3, lambda_bg=0.05, lambda_div=0.01, lambda_mask=0.001):
    q_fg = aux['q_fg']
    q_bg = aux['q_bg']
    fg_protos = aux['fg_protos']
    bg_protos = aux['bg_protos']
    query_mask = aux['query_mask']

    NQ, N = q_fg.shape[0], fg_protos.shape[0]
    dist_fg_all = 1 - F.cosine_similarity(q_fg.unsqueeze(1), fg_protos.unsqueeze(0), dim=-1)
    dist_bg_proto = 1 - F.cosine_similarity(q_fg.unsqueeze(1), bg_protos.unsqueeze(0), dim=-1)
    dist_qbg_bg = 1 - F.cosine_similarity(q_bg.unsqueeze(1), bg_protos.unsqueeze(0), dim=-1)

    loss_triplet = 0.0
    loss_bg = 0.0
    for i in range(NQ):
        y = labels[i]
        mask = torch.ones(N, dtype=torch.bool, device=q_fg.device)
        mask[y] = False
        d_pos_fg = dist_fg_all[i, y]
        d_neg_fg = dist_fg_all[i, mask].min()
        loss_triplet = loss_triplet + torch.clamp(margin + d_pos_fg - d_neg_fg, min=0.0)

        d_pos_bg = dist_bg_proto[i, y]
        d_qbg_bg = dist_qbg_bg[i, y]
        loss_bg = loss_bg + torch.clamp(margin + d_pos_fg - d_pos_bg, min=0.0) + 0.5 * d_qbg_bg

    loss_triplet = loss_triplet / NQ
    loss_bg = loss_bg / NQ

    fg_n = F.normalize(fg_protos, dim=-1)
    sim_mat = torch.matmul(fg_n, fg_n.t())
    eye = torch.eye(sim_mat.size(0), device=sim_mat.device)
    loss_div = ((sim_mat - eye) ** 2).mean()
    loss_mask = query_mask.mean()

    total_aux = loss_triplet + lambda_bg * loss_bg + lambda_div * loss_div + lambda_mask * loss_mask
    return total_aux, {
        'triplet': loss_triplet,
        'bg': loss_bg,
        'div': loss_div,
        'mask': loss_mask,
    }


def train(train_loader, model, criterion, optimizer, epoch_index, F_txt):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.train()
    end = time.time()

    for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_var1 = torch.cat(query_images, 0).cuda()
        input_var2 = torch.cat(support_images, 0).squeeze(0).cuda()
        input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3), input_var2.size(4))
        target = torch.cat(query_targets, 0).cuda()

        logits, aux = model(input_var1, input_var2)
        loss_cls = criterion(logits, target)
        loss_aux, loss_dict = compute_v2_losses(aux, target, opt.margin, opt.lambda_bg, opt.lambda_div, opt.lambda_mask)
        loss = loss_cls + opt.eta * loss_aux

        prec1, _ = utils.accuracy(logits, target, topk=(1, 3))
        losses.update(loss.item(), target.size(0))
        top1.update(prec1[0], target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if episode_index % opt.print_freq == 0 and episode_index != 0:
            msg = ('Epoch-({0}): [{1}/{2}]\t'
                   'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                   'Loss {loss.val:.2f} ({loss.avg:.2f})\t'
                   'L_cls {l_cls:.4f} L_aux {l_aux:.4f} | trip {trip:.4f} bg {bg:.4f} div {div:.4f} mask {mask:.4f}\t'
                   'Prec@1 {top1.val:.2f} ({top1.avg:.2f})'.format(
                epoch_index, episode_index, len(train_loader),
                batch_time=batch_time, loss=losses,
                top1=top1, l_cls=loss_cls.item(), l_aux=loss_aux.item(),
                trip=loss_dict['triplet'].item(), bg=loss_dict['bg'].item(),
                div=loss_dict['div'].item(), mask=loss_dict['mask'].item()))
            print(msg)
            if F_txt:
                print(msg, file=F_txt)

    return top1.avg, losses.avg


def validate(val_loader, model, criterion, epoch_index, best_prec1, F_txt):
    batch_time = utils.AverageMeter()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    model.eval()
    accuracies = []
    end = time.time()

    for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(val_loader):
        input_var1 = torch.cat(query_images, 0).cuda()
        input_var2 = torch.cat(support_images, 0).squeeze(0).cuda()
        input_var2 = input_var2.contiguous().view(-1, input_var2.size(2), input_var2.size(3), input_var2.size(4))
        target = torch.cat(query_targets, 0).cuda()

        logits, _ = model(input_var1, input_var2)
        loss = criterion(logits, target)

        prec1, _ = utils.accuracy(logits, target, topk=(1, 3))
        losses.update(loss.item(), target.size(0))
        top1.update(prec1[0], target.size(0))
        accuracies.append(prec1)

        batch_time.update(time.time() - end)
        end = time.time()

        if episode_index % opt.print_freq == 0 and episode_index != 0:
            msg = ('Val-({0}): [{1}/{2}]\t'
                   'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                   'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch_index, episode_index, len(val_loader),
                batch_time=batch_time, loss=losses, top1=top1))
            print(msg)
            if F_txt:
                print(msg, file=F_txt)

    msg = ' * Prec@1 {top1.avg:.3f} Best_prec1 {best_prec1:.3f}'.format(top1=top1, best_prec1=best_prec1)
    print(msg)
    if F_txt:
        print(msg, file=F_txt)

    return top1.avg, losses.avg, accuracies


if __name__ == '__main__':
    opt.outf, F_txt = utils.set_save_path(opt)
    global best_prec1
    best_prec1 = 0

    model = SlotSAFNet(
        encoder_model=opt.encoder_model,
        way_num=opt.way_num,
        shot_num=opt.shot_num,
        query_num=opt.query_num,
        num_slots=opt.num_slots,
        num_iters=opt.num_iters,
        beta=opt.beta,
        eta=opt.eta,
        margin=opt.margin,
        query_seed_mode=opt.query_seed_mode,
        use_bg_logits=opt.use_bg_logits,
    )

    if opt.pretrained_encoder and os.path.isfile(opt.pretrained_encoder):
        print(f'[INFO] Loading pretrained encoder from: {opt.pretrained_encoder}')
        if F_txt:
            print(f'[INFO] Loading pretrained encoder from: {opt.pretrained_encoder}', file=F_txt)
        ckpt = torch.load(opt.pretrained_encoder, map_location='cpu')
        pretrained_dict = ckpt.get('encoder_state_dict', ckpt.get('model', {}))

        model_dict = model.state_dict()
        matched, unmatched = {}, []
        for k, v in pretrained_dict.items():
            new_key = 'encoder.' + k if not k.startswith('encoder.') else k
            if new_key in model_dict and model_dict[new_key].shape == v.shape:
                matched[new_key] = v
            else:
                unmatched.append(k)

        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f'[INFO] Loaded {len(matched)} encoder tensors. Unmatched: {len(unmatched)}')
        if F_txt:
            print(f'[INFO] Loaded {len(matched)} encoder tensors. Unmatched: {len(unmatched)}', file=F_txt)
    elif opt.pretrained_encoder:
        print(f'[WARNING] pretrained_encoder not found: {opt.pretrained_encoder}')

    if opt.cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    backbone_lr = opt.lr * 0.05 if (opt.pretrained_encoder and os.path.isfile(opt.pretrained_encoder)) else opt.lr
    encoder_params = list(model.encoder.parameters())
    encoder_param_ids = set(id(p) for p in encoder_params)
    other_params = [p for p in model.parameters() if id(p) not in encoder_param_ids]
    param_groups = [
        {'params': encoder_params, 'lr': backbone_lr},
        {'params': other_params, 'lr': opt.lr},
    ]
    msg_lr = f'backbone_lr={backbone_lr:.6f}, other_lr={opt.lr:.6f}'

    if opt.adam:
        optimizer = optim.Adam(param_groups, betas=(opt.beta1, 0.9), weight_decay=opt.weight_decay)
        print(f'Using Adam Optimizer | {msg_lr}')
        if F_txt:
            print(f'Using Adam Optimizer | {msg_lr}', file=F_txt)
    else:
        optimizer = optim.SGD(param_groups, momentum=0.9, dampening=0.0, weight_decay=opt.weight_decay)
        print(f'Using SGD Optimizer | {msg_lr}')
        if F_txt:
            print(f'Using SGD Optimizer | {msg_lr}', file=F_txt)

    if opt.resume:
        checkpoint = utils.get_resume_file(opt.resume, F_txt)
        opt.start_epoch = checkpoint['epoch_index']
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if opt.ngpu > 1:
        model = nn.DataParallel(model, range(opt.ngpu))

    print(opt)
    if F_txt:
        print(opt, file=F_txt)
    print(model)
    if F_txt:
        print(model, file=F_txt)

    if opt.cosine:
        eta_min = opt.lr * (opt.lr_decay_rate ** 3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.epochs, eta_min, -1)

    print('===================================== Training on the train set =====================================')
    if F_txt:
        print('===================================== Training on the train set =====================================', file=F_txt)

    Train_losses, Val_losses, Test_losses = [], [], []

    for epoch_item in range(opt.start_epoch, opt.epochs):
        print('==================== Epoch %d ====================' % epoch_item)
        if F_txt:
            print('==================== Epoch %d ====================', file=F_txt)

        opt.current_epoch = epoch_item
        train_loader, val_loader, test_loader = FewShotDataloader.get_Fewshot_dataloader(opt, ['train', 'val', 'test'])

        prec1_train, train_loss = train(train_loader, model, criterion, optimizer, epoch_item, F_txt)
        Train_losses.append(train_loss)

        print('===================================== Validation on the val set =====================================')
        if F_txt:
            print('===================================== Validation on the val set =====================================', file=F_txt)
        with torch.no_grad():
            prec1_val, val_loss, _ = validate(val_loader, model, criterion, epoch_item, best_prec1, F_txt)
        Val_losses.append(val_loss)

        print('===================================== Validation on the test set =====================================')
        if F_txt:
            print('===================================== Validation on the test set =====================================', file=F_txt)
        with torch.no_grad():
            prec1_test, test_loss, _ = validate(test_loader, model, criterion, epoch_item, best_prec1, F_txt)
        Test_losses.append(test_loss)

        if opt.cosine:
            scheduler.step()
        else:
            utils.adjust_learning_rate(opt, optimizer, epoch_item, F_txt)

        is_best = prec1_val > best_prec1
        best_prec1 = max(prec1_val, best_prec1)

        if is_best:
            utils.save_checkpoint({
                'epoch_index': epoch_item,
                'encoder_model': opt.encoder_model,
                'classifier_model': opt.classifier_model,
                'model': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict(),
            }, os.path.join(opt.outf, 'model_best.pth.tar'))

        utils.save_checkpoint({
            'epoch_index': epoch_item,
            'encoder_model': opt.encoder_model,
            'classifier_model': opt.classifier_model,
            'model': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, os.path.join(opt.outf, f'epoch_{epoch_item}.pth.tar'))

    utils.plot_loss_curve(opt, Train_losses, Val_losses, Test_losses)
    print('======================================== Training is END ========================================\n')
    if F_txt:
        print('======================================== Training is END ========================================\n', file=F_txt)
    F_txt.close()
