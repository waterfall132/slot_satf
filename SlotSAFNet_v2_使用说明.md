# SlotSAFNet v2 使用说明

本文档对应当前目录 `E:\slot_satf` 中已经实现的 SlotSAFNet v2 代码。

---

## 1. 这版 v2 改了什么

相对之前版本，当前 v2 已经实现：

1. **更强的 Slot 模块**
   - q/k/v learnable projection
   - residual + MLP update
   - LayerNorm
   - support prototype-seeded / query self-seeded / global seed

2. **query 端真正改成 class-agnostic**
   - 默认使用 `self` 模式，即 query 自己的 descriptor 均值作为 seed
   - 不再依赖 support prototype average 去引导 query

3. **显式 foreground / background prototypes**
   - support 端同时构造 `fg_protos` 与 `bg_protos`

4. **更完整的 logits**
   - foreground matching
   - foreground-to-background separation
   - query background to background prototype consistency

5. **更完整的辅助损失**
   - foreground hardest-negative triplet
   - background separation loss
   - prototype diversity loss
   - query mask regularization

---

## 2. 关键文件

- 模型：
  - `models/slot_saf_net.py`
- few-shot 训练：
  - `Train_SlotSAF.py`
- few-shot 测试：
  - `Test_SlotSAF.py`
- backbone 预训练：
  - `Pretrain_ResNet12.py`

---

## 3. 推荐实验流程

建议严格按以下顺序做：

### Step 1：预训练 backbone
先跑 `Pretrain_ResNet12.py`，得到 `pretrain_best.pth.tar`。

### Step 2：few-shot 训练
用预训练权重训练 SlotSAFNet v2。

### Step 3：few-shot 测试
用 `Test_SlotSAF.py` 对 best checkpoint 做多轮测试。

---

## 4. 预训练命令

## 4.1 CUB
```bash
python Pretrain_ResNet12.py ^
  --dataset_dir E:\dataset\CUB_200_2011_FewShot ^
  --data_name CubBird ^
  --encoder_model ResNet12 ^
  --imageSize 84 ^
  --epochs 500 ^
  --batch_size 16 ^
  --lr 0.001 ^
  --lr_decay_epochs 75 150 300 ^
  --outf .\pretrain_results_feat
```

## 4.2 miniImageNet
```bash
python Pretrain_ResNet12.py ^
  --dataset_dir E:\dataset\miniImageNet--ravi ^
  --data_name miniImageNet ^
  --encoder_model ResNet12 ^
  --imageSize 84 ^
  --epochs 100 ^
  --batch_size 128 ^
  --lr 0.1 ^
  --lr_decay_epochs 60 80 ^
  --outf .\pretrain_results_feat
```

## 4.3 tieredImageNet
```bash
python Pretrain_ResNet12.py ^
  --dataset_dir E:\dataset\tieredImageNet ^
  --data_name tieredImageNet ^
  --encoder_model ResNet12 ^
  --imageSize 84 ^
  --epochs 150 ^
  --batch_size 64 ^
  --lr 0.05 ^
  --lr_decay_epochs 60 90 120 ^
  --outf .\pretrain_results_feat
```

## 4.4 Stanford Dog
```bash
python Pretrain_ResNet12.py ^
  --dataset_dir E:\dataset\StanfordDog_FewShot ^
  --data_name StanfordDog ^
  --encoder_model ResNet12 ^
  --imageSize 84 ^
  --epochs 300 ^
  --batch_size 32 ^
  --lr 0.001 ^
  --lr_decay_epochs 75 150 225 ^
  --outf .\pretrain_results_feat
```

## 4.5 Stanford Car
```bash
python Pretrain_ResNet12.py ^
  --dataset_dir E:\dataset\StanfordCar_FewShot ^
  --data_name StanfordCar ^
  --encoder_model ResNet12 ^
  --imageSize 84 ^
  --epochs 300 ^
  --batch_size 32 ^
  --lr 0.001 ^
  --lr_decay_epochs 75 150 225 ^
  --outf .\pretrain_results_feat
```

---

## 5. few-shot 训练命令

下面默认：
- 5-way 1-shot
- ResNet12 backbone
- 使用预训练 encoder
- query seed 使用 `self`

## 5.1 CUB 5-way 1-shot（推荐起点）
```bash
python Train_SlotSAF.py ^
  --dataset_dir E:\dataset\CUB_200_2011_FewShot ^
  --data_name CubBird ^
  --encoder_model ResNet12 ^
  --classifier_model SlotSAFNet ^
  --pretrained_encoder .\pretrain_results_feat\CubBird\pretrain_best.pth.tar ^
  --way_num 5 ^
  --shot_num 1 ^
  --query_num 15 ^
  --episode_train_num 4000 ^
  --episode_val_num 1000 ^
  --episode_test_num 1000 ^
  --epochs 30 ^
  --lr 0.0005 ^
  --num_slots 5 ^
  --num_iters 3 ^
  --eta 0.1 ^
  --lambda_bg 0.05 ^
  --lambda_div 0.01 ^
  --lambda_mask 0.001 ^
  --margin 0.3 ^
  --query_seed_mode self ^
  --use_bg_logits ^
  --outf .\results_SlotSAFNet_v2_CUB_5way1shot
```

## 5.2 CUB 5-way 5-shot
```bash
python Train_SlotSAF.py ^
  --dataset_dir E:\dataset\CUB_200_2011_FewShot ^
  --data_name CubBird ^
  --encoder_model ResNet12 ^
  --classifier_model SlotSAFNet ^
  --pretrained_encoder .\pretrain_results_feat\CubBird\pretrain_best.pth.tar ^
  --way_num 5 ^
  --shot_num 5 ^
  --query_num 15 ^
  --episode_train_num 4000 ^
  --episode_val_num 1000 ^
  --episode_test_num 1000 ^
  --epochs 30 ^
  --lr 0.0005 ^
  --num_slots 5 ^
  --num_iters 3 ^
  --eta 0.1 ^
  --lambda_bg 0.05 ^
  --lambda_div 0.01 ^
  --lambda_mask 0.001 ^
  --margin 0.3 ^
  --query_seed_mode self ^
  --use_bg_logits ^
  --outf .\results_SlotSAFNet_v2_CUB_5way5shot
```

## 5.3 miniImageNet 5-way 1-shot
```bash
python Train_SlotSAF.py ^
  --dataset_dir E:\dataset\miniImageNet--ravi ^
  --data_name miniImageNet ^
  --encoder_model ResNet12 ^
  --classifier_model SlotSAFNet ^
  --pretrained_encoder .\pretrain_results_feat\miniImageNet\pretrain_best.pth.tar ^
  --way_num 5 ^
  --shot_num 1 ^
  --query_num 15 ^
  --episode_train_num 6000 ^
  --episode_val_num 1000 ^
  --episode_test_num 1000 ^
  --epochs 40 ^
  --lr 0.0005 ^
  --num_slots 5 ^
  --num_iters 3 ^
  --eta 0.1 ^
  --lambda_bg 0.05 ^
  --lambda_div 0.01 ^
  --lambda_mask 0.001 ^
  --margin 0.3 ^
  --query_seed_mode self ^
  --use_bg_logits ^
  --outf .\results_SlotSAFNet_v2_mini_5way1shot
```

## 5.4 miniImageNet 5-way 5-shot
```bash
python Train_SlotSAF.py ^
  --dataset_dir E:\dataset\miniImageNet--ravi ^
  --data_name miniImageNet ^
  --encoder_model ResNet12 ^
  --classifier_model SlotSAFNet ^
  --pretrained_encoder .\pretrain_results_feat\miniImageNet\pretrain_best.pth.tar ^
  --way_num 5 ^
  --shot_num 5 ^
  --query_num 15 ^
  --episode_train_num 6000 ^
  --episode_val_num 1000 ^
  --episode_test_num 1000 ^
  --epochs 40 ^
  --lr 0.0005 ^
  --num_slots 5 ^
  --num_iters 3 ^
  --eta 0.1 ^
  --lambda_bg 0.05 ^
  --lambda_div 0.01 ^
  --lambda_mask 0.001 ^
  --margin 0.3 ^
  --query_seed_mode self ^
  --use_bg_logits ^
  --outf .\results_SlotSAFNet_v2_mini_5way5shot
```

## 5.5 tieredImageNet 5-way 1-shot
```bash
python Train_SlotSAF.py ^
  --dataset_dir E:\dataset\tieredImageNet ^
  --data_name tieredImageNet ^
  --encoder_model ResNet12 ^
  --classifier_model SlotSAFNet ^
  --pretrained_encoder .\pretrain_results_feat\tieredImageNet\pretrain_best.pth.tar ^
  --way_num 5 ^
  --shot_num 1 ^
  --query_num 15 ^
  --episode_train_num 6000 ^
  --episode_val_num 1000 ^
  --episode_test_num 1000 ^
  --epochs 40 ^
  --lr 0.0005 ^
  --num_slots 5 ^
  --num_iters 3 ^
  --eta 0.1 ^
  --lambda_bg 0.05 ^
  --lambda_div 0.01 ^
  --lambda_mask 0.001 ^
  --margin 0.3 ^
  --query_seed_mode self ^
  --use_bg_logits ^
  --outf .\results_SlotSAFNet_v2_tiered_5way1shot
```

---

## 6. 测试命令

训练结束后会自动测试，但你也可以手动测试。

### 6.1 CUB 手动测试
```bash
python Test_SlotSAF.py ^
  --dataset_dir E:\dataset\CUB_200_2011_FewShot ^
  --data_name CubBird ^
  --encoder_model ResNet12 ^
  --way_num 5 ^
  --shot_num 1 ^
  --query_num 15 ^
  --num_slots 5 ^
  --num_iters 3 ^
  --eta 0.1 ^
  --margin 0.3 ^
  --query_seed_mode self ^
  --use_bg_logits ^
  --resume .\results_SlotSAFNet_v2_CUB_5way1shot
```

### 6.2 miniImageNet 手动测试
```bash
python Test_SlotSAF.py ^
  --dataset_dir E:\dataset\miniImageNet--ravi ^
  --data_name miniImageNet ^
  --encoder_model ResNet12 ^
  --way_num 5 ^
  --shot_num 1 ^
  --query_num 15 ^
  --num_slots 5 ^
  --num_iters 3 ^
  --eta 0.1 ^
  --margin 0.3 ^
  --query_seed_mode self ^
  --use_bg_logits ^
  --resume .\results_SlotSAFNet_v2_mini_5way1shot
```

---

## 7. 关键超参数说明

## 7.1 结构超参数

### `--num_slots`
slot 数量。

推荐：
- CUB 1-shot：4~6
- miniImageNet：4~6
- 细粒度任务可先试 5

### `--num_iters`
slot 迭代轮数。

推荐：
- 起步 3
- 如果不稳定，可减到 2
- 如果 mask 过粗，可试 4~5

### `--query_seed_mode`
query seed 方式：
- `self`：query 自身 descriptor 平均，推荐默认
- `global`：使用全局 learnable slot seed

推荐：
- 先用 `self`
- 如果 query mask 太不稳定，再试 `global`

---

## 7.2 loss 超参数

### `--eta`
辅助损失总权重。

推荐搜索：
- 0.05
- 0.1
- 0.2

### `--lambda_bg`
背景分离损失权重。

推荐搜索：
- 0.02
- 0.05
- 0.1

### `--lambda_div`
prototype diversity loss 权重。

推荐搜索：
- 0.005
- 0.01
- 0.02

### `--lambda_mask`
query mask regularization 权重。

推荐搜索：
- 0.0005
- 0.001
- 0.005

### `--margin`
triplet / bg separation margin。

推荐搜索：
- 0.1
- 0.2
- 0.3
- 0.5

CUB 上建议先从 `0.3` 起。

---

## 7.3 训练超参数

### `--lr`
few-shot 阶段主学习率。

推荐：
- ResNet12 + pretrained：`5e-4`
- 如果不稳定：`1e-4 ~ 3e-4`

### `--episode_train_num`
每个 epoch 的 train episodes 数。

推荐：
- CUB：4000
- miniImageNet：6000
- 如果过拟合严重，优先降低，而不是盲目加 epoch

### `--epochs`
推荐：
- CUB：30
- miniImageNet / tiered：40

---

## 8. 推荐调参顺序

不要一上来全调，建议顺序如下：

### 第一阶段：结构固定，只调 loss
固定：
- `num_slots=5`
- `num_iters=3`
- `query_seed_mode=self`

只搜索：
- `eta`
- `lambda_bg`
- `margin`

### 第二阶段：调 slot 结构
搜索：
- `num_slots`
- `num_iters`
- `query_seed_mode`

### 第三阶段：调正则
搜索：
- `lambda_div`
- `lambda_mask`

---

## 9. 建议的消融命令

## 9.1 去掉背景 logits
```bash
python Train_SlotSAF.py ^
  --dataset_dir E:\dataset\CUB_200_2011_FewShot ^
  --data_name CubBird ^
  --encoder_model ResNet12 ^
  --classifier_model SlotSAFNet ^
  --pretrained_encoder .\pretrain_results_feat\CubBird\pretrain_best.pth.tar ^
  --way_num 5 --shot_num 1 --query_num 15 ^
  --episode_train_num 4000 --episode_val_num 1000 --episode_test_num 1000 ^
  --epochs 30 --lr 0.0005 ^
  --num_slots 5 --num_iters 3 ^
  --eta 0.1 --lambda_bg 0.05 --lambda_div 0.01 --lambda_mask 0.001 ^
  --margin 0.3 --query_seed_mode self ^
  --outf .\ablation_no_bg_logits
```

## 9.2 query global seed
```bash
python Train_SlotSAF.py ^
  --dataset_dir E:\dataset\CUB_200_2011_FewShot ^
  --data_name CubBird ^
  --encoder_model ResNet12 ^
  --classifier_model SlotSAFNet ^
  --pretrained_encoder .\pretrain_results_feat\CubBird\pretrain_best.pth.tar ^
  --way_num 5 --shot_num 1 --query_num 15 ^
  --episode_train_num 4000 --episode_val_num 1000 --episode_test_num 1000 ^
  --epochs 30 --lr 0.0005 ^
  --num_slots 5 --num_iters 3 ^
  --eta 0.1 --lambda_bg 0.05 --lambda_div 0.01 --lambda_mask 0.001 ^
  --margin 0.3 --query_seed_mode global --use_bg_logits ^
  --outf .\ablation_query_global_seed
```

## 9.3 fewer slots
```bash
python Train_SlotSAF.py ^
  --dataset_dir E:\dataset\CUB_200_2011_FewShot ^
  --data_name CubBird ^
  --encoder_model ResNet12 ^
  --classifier_model SlotSAFNet ^
  --pretrained_encoder .\pretrain_results_feat\CubBird\pretrain_best.pth.tar ^
  --way_num 5 --shot_num 1 --query_num 15 ^
  --episode_train_num 4000 --episode_val_num 1000 --episode_test_num 1000 ^
  --epochs 30 --lr 0.0005 ^
  --num_slots 3 --num_iters 3 ^
  --eta 0.1 --lambda_bg 0.05 --lambda_div 0.01 --lambda_mask 0.001 ^
  --margin 0.3 --query_seed_mode self --use_bg_logits ^
  --outf .\ablation_slots3
```

---

## 10. 使用建议总结

如果你现在只想先跑一个最稳的版本，推荐直接用：

- dataset: CUB
- setting: 5-way 1-shot
- `num_slots=5`
- `num_iters=3`
- `query_seed_mode=self`
- `eta=0.1`
- `lambda_bg=0.05`
- `lambda_div=0.01`
- `lambda_mask=0.001`
- `margin=0.3`
- `episode_train_num=4000`
- `lr=5e-4`

如果这个版本比旧版仍无明显提升，再优先做两组消融：

1. `query_seed_mode=self` vs `global`
2. `use_bg_logits` on vs off

这两组最能快速判断 v2 的关键设计是否有效。
