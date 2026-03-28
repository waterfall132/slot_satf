#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone import ResNet12, Conv64F_Local, Conv64F

ENCODER_FEAT_DIM = {
    'ResNet12': 640,
    'Conv64F_Local': 64,
    'Conv4': 64,
    'Conv64F': 64,
}


def build_encoder(encoder_model: str) -> nn.Module:
    if encoder_model == 'ResNet12':
        return ResNet12(keep_prob=1.0, avg_pool=False, flatten=False)
    elif encoder_model == 'Conv64F_Local':
        return Conv64F_Local()
    elif encoder_model in ['Conv4', 'Conv64F']:
        return Conv64F()
    else:
        raise ValueError(f"Unsupported encoder: {encoder_model}")


class SlotAttentionV2(nn.Module):
    def __init__(self, feat_dim: int, num_slots: int = 5, num_iters: int = 3, eps: float = 1e-8):
        super().__init__()
        self.V = num_slots
        self.T = num_iters
        self.eps = eps
        self.feat_dim = feat_dim

        self.to_q = nn.Linear(feat_dim, feat_dim, bias=False)
        self.to_k = nn.Linear(feat_dim, feat_dim, bias=False)
        self.to_v = nn.Linear(feat_dim, feat_dim, bias=False)

        self.scale = nn.Parameter(torch.tensor(1.0 / math.sqrt(feat_dim), dtype=torch.float32))
        self.norm_in = nn.LayerNorm(feat_dim)
        self.norm_slots = nn.LayerNorm(feat_dim)
        self.norm_update = nn.LayerNorm(feat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim * 2, feat_dim)
        )

        self.global_slot_seed = nn.Parameter(torch.randn(1, num_slots, feat_dim) * 0.02)
        self.slot_noise_scale = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def init_slots(self, features: torch.Tensor, seed: torch.Tensor = None, mode: str = 'self'):
        B, _, C = features.shape
        if mode == 'prototype' and seed is not None:
            base = seed.unsqueeze(1).expand(-1, self.V, -1)
        elif mode == 'global':
            base = self.global_slot_seed.expand(B, -1, -1)
        else:
            self_seed = features.mean(dim=1, keepdim=True)
            base = self_seed.expand(-1, self.V, -1)

        noise = torch.randn_like(base) * self.slot_noise_scale.abs()
        return base + noise

    def forward(self, features: torch.Tensor, seed: torch.Tensor = None, mode: str = 'self'):
        x = self.norm_in(features)
        q = self.to_q(x)
        v = self.to_v(x)

        slots = self.init_slots(features, seed=seed, mode=mode)
        attn_maps = None

        for _ in range(self.T):
            slots_norm = self.norm_slots(slots)
            k = self.to_k(slots_norm)
            attn_logits = torch.einsum('blc,bvc->blv', q, k) * self.scale
            attn = F.softmax(attn_logits, dim=-1)
            attn_sum = attn.sum(dim=1, keepdim=True) + self.eps
            updates = torch.einsum('blv,blc->bvc', attn, v) / attn_sum.transpose(1, 2)
            slots = slots + self.mlp(self.norm_update(updates))
            attn_maps = attn

        return slots, attn_maps


def compute_soft_mask(slots: torch.Tensor, attn_maps: torch.Tensor, prototype: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    slots_n = F.normalize(slots, dim=-1, eps=eps)
    proto_n = F.normalize(prototype, dim=-1, eps=eps).unsqueeze(1)
    alpha = ((slots_n * proto_n).sum(dim=-1) + 1.0) / 2.0
    soft_mask = (attn_maps * alpha.unsqueeze(1)).sum(dim=-1)
    return soft_mask.clamp(0.0, 1.0)


def decouple_and_pool(features: torch.Tensor, soft_mask: torch.Tensor, eps: float = 1e-8):
    fg_w = soft_mask.unsqueeze(-1)
    bg_w = (1.0 - soft_mask).unsqueeze(-1)
    fg_feat_sum = (fg_w * features).sum(dim=1)
    bg_feat_sum = (bg_w * features).sum(dim=1)
    fg_wsum = fg_w.sum(dim=1) + eps
    bg_wsum = bg_w.sum(dim=1) + eps
    fg_vec = fg_feat_sum / fg_wsum
    bg_vec = bg_feat_sum / bg_wsum
    return fg_feat_sum, fg_wsum, bg_feat_sum, bg_wsum, fg_vec, bg_vec, soft_mask


class SlotSAFNet(nn.Module):
    def __init__(self, encoder_model='ResNet12', way_num=5, shot_num=1, query_num=15,
                 num_slots=5, num_iters=3, eta=0.1, margin=0.3, beta=10.0, lam=1.0,
                 query_seed_mode='self', use_bg_logits=True):
        super().__init__()
        self.N = way_num
        self.K = shot_num
        self.Q = query_num
        self.eta = eta
        self.margin = margin
        self.eps = 1e-8
        self.query_seed_mode = query_seed_mode
        self.use_bg_logits = use_bg_logits

        self.encoder = build_encoder(encoder_model)
        feat_dim = ENCODER_FEAT_DIM[encoder_model]
        self.feat_dim = feat_dim

        self.slot_attn = SlotAttentionV2(feat_dim=feat_dim, num_slots=num_slots, num_iters=num_iters, eps=self.eps)

        self.gamma_fg = nn.Parameter(torch.tensor(float(beta)))
        self.gamma_bg = nn.Parameter(torch.tensor(float(beta) * 0.5))
        self.beta_bg = nn.Parameter(torch.tensor(0.5))

    def extract(self, images: torch.Tensor) -> torch.Tensor:
        feat_map = self.encoder(images)
        B, C, h, w = feat_map.shape
        return feat_map.view(B, C, h * w).permute(0, 2, 1)

    def _run_slot_saf(self, feat: torch.Tensor, proto: torch.Tensor = None, mode: str = 'self'):
        if proto is None:
            proto = feat.mean(dim=1)
        slots, attn_maps = self.slot_attn(feat, seed=proto, mode=mode)
        soft_mask = compute_soft_mask(slots, attn_maps, proto, self.eps)
        return decouple_and_pool(feat, soft_mask, self.eps)

    def _build_support_prototypes(self, sup_feat: torch.Tensor, coarse_protos: torch.Tensor):
        fg_list, bg_list, masks = [], [], []
        for n in range(self.N):
            proto_n_expand = coarse_protos[n].unsqueeze(0).expand(self.K, -1)
            fg_sum, fg_w, bg_sum, bg_w, _, _, soft_mask = self._run_slot_saf(sup_feat[n], proto_n_expand, mode='prototype')
            p_fg = fg_sum.sum(dim=0) / (fg_w.sum(dim=0) + self.eps)
            p_bg = bg_sum.sum(dim=0) / (bg_w.sum(dim=0) + self.eps)
            fg_list.append(p_fg)
            bg_list.append(p_bg)
            masks.append(soft_mask)
        return torch.stack(fg_list, dim=0), torch.stack(bg_list, dim=0), masks

    def _build_query_repr(self, q_feat: torch.Tensor):
        if self.query_seed_mode == 'global':
            proto = q_feat.mean(dim=1)
            return self._run_slot_saf(q_feat, proto, mode='global')
        else:
            proto = q_feat.mean(dim=1)
            return self._run_slot_saf(q_feat, proto, mode='self')

    def forward(self, query_images: torch.Tensor, support_images: torch.Tensor):
        N, K = self.N, self.K
        _, C_in, H_in, W_in = support_images.shape
        support_images = support_images.view(N, K, C_in, H_in, W_in)

        sup_flat = support_images.view(N * K, C_in, H_in, W_in)
        sup_feat = self.extract(sup_flat)
        _, L, C = sup_feat.shape
        sup_feat = sup_feat.view(N, K, L, C)
        q_feat = self.extract(query_images)

        coarse_protos = sup_feat.mean(dim=[1, 2])
        fg_protos, bg_protos, support_masks = self._build_support_prototypes(sup_feat, coarse_protos)

        q_fg_sum, q_fg_w, q_bg_sum, q_bg_w, q_fg, q_bg, query_mask = self._build_query_repr(q_feat)

        q_fg_n = F.normalize(q_fg + self.eps, dim=-1)
        q_bg_n = F.normalize(q_bg + self.eps, dim=-1)
        fg_n = F.normalize(fg_protos + self.eps, dim=-1)
        bg_n = F.normalize(bg_protos + self.eps, dim=-1)

        dist_fg = 1.0 - (q_fg_n.unsqueeze(1) * fg_n.unsqueeze(0)).sum(dim=-1)
        dist_bg_sep = 1.0 - (q_fg_n.unsqueeze(1) * bg_n.unsqueeze(0)).sum(dim=-1)
        dist_qbg_bg = 1.0 - (q_bg_n.unsqueeze(1) * bg_n.unsqueeze(0)).sum(dim=-1)

        logits_fg = -self.gamma_fg * dist_fg
        if self.use_bg_logits:
            logits_bg = self.gamma_bg * dist_bg_sep - self.beta_bg * dist_qbg_bg
            logits = logits_fg + logits_bg
        else:
            logits = logits_fg

        aux = {
            'fg_protos': fg_protos,
            'bg_protos': bg_protos,
            'q_fg': q_fg,
            'q_bg': q_bg,
            'dist_fg': dist_fg,
            'dist_bg_sep': dist_bg_sep,
            'dist_qbg_bg': dist_qbg_bg,
            'support_masks': support_masks,
            'query_mask': query_mask,
        }
        return logits, aux
