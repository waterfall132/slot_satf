#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import time
from PIL import ImageFile
import sys

sys.dont_write_bytecode = True

import dataset.general_dataloader as FewShotDataloader
import utils
from models.slot_saf_net import SlotSAFNet

ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', default='/root/autodl-tmp/CUB200/CUB_200_2011_FewShot')
parser.add_argument('--data_name', default='CubBird')
parser.add_argument('--mode', default='test')
parser.add_argument('--outf', default='')
parser.add_argument('--resume', default='', type=str, help='checkpoint 文件路径或目录（目录时自动找 model_best.pth.tar）')
parser.add_argument('--encoder_model', default='ResNet12')
parser.add_argument('--classifier_model', default='SlotSAFNet')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--imageSize', type=int, default=84)
parser.add_argument('--train_aug', action='store_true', default=False)
parser.add_argument('--test_aug', action='store_true', default=False)
parser.add_argument('--aug_shot_num', type=int, default=20)
parser.add_argument('--neighbor_k', type=int, default=1)

# Few-shot
parser.add_argument('--episodeSize', type=int, default=1)
parser.add_argument('--testepisodeSize', type=int, default=1)
parser.add_argument('--episode_test_num', type=int, default=1000)
parser.add_argument('--way_num', type=int, default=5)
parser.add_argument('--shot_num', type=int, default=1)
parser.add_argument('--query_num', type=int, default=15)

# SlotSAFNet 参数（必须和训练时一致）
parser.add_argument('--num_slots', type=int, default=5)
parser.add_argument('--num_iters', type=int, default=3)
parser.add_argument('--beta',   type=float, default=10.0)
parser.add_argument('--lam',    type=float, default=1.0)
parser.add_argument('--eta',    type=float, default=0.1)
parser.add_argument('--margin', type=float, default=0.3)
parser.add_argument('--query_seed_mode', type=str, default='self', choices=['self', 'global'])
parser.add_argument('--use_bg_logits', action='store_true', default=True)

parser.add_argument('--cuda', action='store_true', default=True)
parser.add_argument('--print_freq', type=int, default=100)
parser.add_argument('--repeat_num', type=int, default=5, help='重复测试轮数，用于计算置信区间')

opt = parser.parse_args()
cudnn.benchmark = True


def test(test_loader, model, criterion, F_txt):
    batch_time = utils.AverageMeter()
    losses     = utils.AverageMeter()
    top1       = utils.AverageMeter()
    accuracies = []

    model.eval()
    device = next(model.parameters()).device
    end = time.time()

    with torch.no_grad():
        for episode_index, (query_images, query_targets, support_images, support_targets) in enumerate(test_loader):

            # Query: list of tensors → [N*Q, C, H, W]
            if isinstance(query_images, list):
                input_var1 = torch.cat([q.to(device) for q in query_images], dim=0)
            else:
                input_var1 = query_images.to(device)

            # Support: list of [K, C, H, W] → stack → [N, K, C, H, W] → view → [N*K, C, H, W]
            # 与训练脚本保持一致
            support_list = [s.to(device).squeeze(0) for s in support_images]  # 每个 [K, C, H, W]
            input_var2   = torch.cat(support_list, dim=0)                      # [N*K, C, H, W]

            target = torch.cat(query_targets, 0).to(device)

            logits, _ = model(input_var1, input_var2)
            loss = criterion(logits, target)

            prec1, _ = utils.accuracy(logits, target, topk=(1, 3))
            losses.update(loss.item(), target.size(0))
            top1.update(prec1[0], target.size(0))
            accuracies.append(prec1[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if episode_index % opt.print_freq == 0 and episode_index != 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    episode_index, len(test_loader),
                    batch_time=batch_time, loss=losses, top1=top1))

    print(' * Final Prec@1 {top1.avg:.3f}'.format(top1=top1))
    if F_txt:
        print(' * Final Prec@1 {top1.avg:.3f}'.format(top1=top1), file=F_txt)

    return top1.avg, losses.avg, accuracies


if __name__ == '__main__':

    # ── 模型初始化 ──────────────────────────────────────────────────
    print("==> Creating SlotSAFNet model...")
    model = SlotSAFNet(
        encoder_model=opt.encoder_model,
        way_num=opt.way_num,
        shot_num=opt.shot_num,
        query_num=opt.query_num,
        num_slots=opt.num_slots,
        num_iters=opt.num_iters,
        beta=opt.beta,
        lam=opt.lam,
        eta=opt.eta,
        margin=opt.margin,
        query_seed_mode=opt.query_seed_mode,
        use_bg_logits=opt.use_bg_logits,
    )
    if opt.cuda:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    # ── checkpoint 路径 ─────────────────────────────────────────────
    if os.path.isdir(opt.resume):
        resume_path = os.path.join(opt.resume, 'model_best.pth.tar')
    else:
        resume_path = opt.resume

    if not resume_path or not os.path.isfile(resume_path):
        print(f"=> no checkpoint found at '{resume_path}'")
        sys.exit(1)

    # ── 输出目录 ────────────────────────────────────────────────────
    if not opt.outf:
        opt.outf = os.path.dirname(resume_path)
    os.makedirs(opt.outf, exist_ok=True)
    txt_save_path = os.path.join(opt.outf, 'Test_results_SlotSAF.txt')
    F_txt = open(txt_save_path, 'a+')

    # ── 加载权重 ────────────────────────────────────────────────────
    print(f"=> loading checkpoint '{resume_path}'")
    ckpt = torch.load(resume_path, map_location=f'cuda:{torch.cuda.current_device()}')
    epoch_index = ckpt.get('epoch_index', 0)
    state_dict  = ckpt.get('model', ckpt.get('state_dict'))
    state_dict  = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    print(f"=> loaded checkpoint (epoch {epoch_index}, best_prec1={ckpt.get('best_prec1', 'N/A')})")

    # ── 多轮测试 ────────────────────────────────────────────────────
    print(f'Test Model: {resume_path}', file=F_txt)
    all_accs = []

    for r in range(opt.repeat_num):
        print(f'==================== Round {r} ====================')
        print(f'==================== Round {r} ====================', file=F_txt)

        seed = r + int(time.time()) % 10000
        random.seed(seed); np.random.seed(seed)
        torch.manual_seed(seed)
        if opt.cuda: torch.cuda.manual_seed(seed)

        opt.mode = 'test'
        test_loader = FewShotDataloader.get_Fewshot_dataloader(opt, ['test'])[0]

        _, _, accuracies = test(test_loader, model, criterion, F_txt)
        acc, h = utils.mean_confidence_interval(accuracies)
        all_accs.append(acc)

        msg = f'Round {r} Accuracy: {acc:.2f} +/- {h:.2f}'
        print(msg); print(msg, file=F_txt)

    # ── 汇总 ────────────────────────────────────────────────────────
    final_acc = np.mean(all_accs)
    final_std = np.std(all_accs)
    msg = f'\n===== Final Result ({opt.repeat_num} rounds): {final_acc:.2f} +/- {final_std:.2f} =====\n'
    print(msg); print(msg, file=F_txt)
    F_txt.close()