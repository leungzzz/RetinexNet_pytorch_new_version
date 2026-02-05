import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import random
import torch
from PIL import Image
# 假设 utils.py 保持不变
from utils import make_dataset

# class PairDataset(data.Dataset):
#     def __init__(self, opt):
#         super(PairDataset, self).__init__()
#         self.opt = opt
#         # 直接在初始化中完成路径获取
#         self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
#         self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

#         self.A_paths = sorted(make_dataset(self.dir_A))
#         self.B_paths = sorted(make_dataset(self.dir_B))
#         self.A_size = len(self.A_paths)

#         # 仅保留基础转换，动态增强放在 __getitem__
#         self.transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#         ])

#     def __getitem__(self, index):
#         A_path = self.A_paths[index % self.A_size]
#         B_path = self.B_paths[index % self.A_size] # 假设 A 和 B 是一一对应的

#         A_img = Image.open(A_path).convert('RGB')
#         B_img = Image.open(B_path).convert('RGB')

#         # 1. 基础转换
#         A_img = self.transform(A_img)
#         B_img = self.transform(B_img)

#         # 2. 数据增强 (Flip)
#         if not self.opt.no_flip:
#             if random.random() < 0.5:
#                 A_img = F.hflip(A_img)
#                 B_img = F.hflip(B_img)
#             if random.random() < 0.5:
#                 A_img = F.vflip(A_img)
#                 B_img = F.vflip(B_img)

#         # 3. 亮度调整 (input_img 逻辑)
#         input_img = A_img
#         if hasattr(self.opt, 'low_times') and random.random() < 0.5:
#             times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
#             # 将归一化后的数据转回 0-1 进行亮度模拟，再转回去
#             input_img = ((A_img + 1.0) / 2.0 / (times + 1e-5)) * 2.0 - 1.0

#         # 4. 计算灰度图 (用于平滑损失权重)
#         # 使用标准的亮度公式，保持张量维度 [1, H, W]
#         # 注意：这里的数据是归一化过的 (-1, 1)
#         r, g, b = (input_img[0:1] + 1), (input_img[1:2] + 1), (input_img[2:3] + 1)
#         A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.

#         return {
#             'A': A_img, 
#             'B': B_img, 
#             'A_gray': A_gray, 
#             'input_img': input_img,
#             'A_paths': A_path, 
#             'B_paths': B_path
#         }

#     def __len__(self):
#         return self.A_size



class PairDataset(data.Dataset):
    def __init__(self, opt):
        super(PairDataset, self).__init__()
        self.opt = opt
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)

        # 改动点：只使用 ToTensor，将像素保持在 [0, 1]
        self.transform = transforms.Compose([
            transforms.ToTensor(), 
        ])

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        # 数据增强 (Flip)
        if not self.opt.no_flip:
            if random.random() < 0.5:
                A_img = F.hflip(A_img); B_img = F.hflip(B_img)
            if random.random() < 0.5:
                A_img = F.vflip(A_img); B_img = F.vflip(B_img)

        # 改动点：亮度调整逻辑 (由于 A_img 已经是 [0, 1])
        input_img = A_img
        if hasattr(self.opt, 'low_times') and random.random() < 0.5:
            times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
            # 直接在 [0, 1] 空间缩放，限制最大值为 1.0
            input_img = torch.clamp(A_img / (times + 1e-5), 0.0, 1.0)

        # 改动点：灰度图计算 (基于 [0, 1] 的 input_img)
        r, g, b = input_img[0:1], input_img[1:2], input_img[2:3]
        # 使用标准心理学公式计算亮度
        A_gray = 0.299 * r + 0.587 * g + 0.114 * b

        return {
            'A': A_img, 
            'B': B_img, 
            'A_gray': A_gray, 
            'input_img': input_img,
            'A_paths': A_path, 
            'B_paths': B_path
        }


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def Get_paired_dataset(batch_size):
    # 更新配置参数
    opt = Config(
        dataroot="/home/mingmou/Downloads/retinexnet_datasets/final_dataset", 
        phase="train", 
        resize_or_crop='no', 
        no_flip=False, # 训练建议开启 flip
        low_times=50, 
        high_times=150
    )
    
    dataset = PairDataset(opt)
    
    # 显著提升速度的关键：num_workers 和 pin_memory
    dataloader = data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True
    )
    return dataloader