import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import os
import random
import torch
from PIL import Image
# 假设 utils.py 保持不变
from utils import make_dataset

# for LUSI dataset
# since the dataset scale is different, we need to crop/resize it. (To least_size*least_size here)
class PairDataset(data.Dataset):
    def __init__(self, opt):
        super(PairDataset, self).__init__()
        self.opt = opt
        self.least_size = 256  # 变为 256*256 的输入/输出尺寸
        
        self.dir_A = os.path.join(opt.dataroot, "low")    # input low light
        self.dir_B = os.path.join(opt.dataroot, "high")   # gt

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)

        # 仅保留基础转换，动态增强放在 __getitem__
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size] # A 和 B 同名，一一对应的

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # --- 修改部分：尺寸检查与对齐 ---
        w, h = A_img.size
        if w < self.least_size or h < self.least_size:
            # 如果有任何一边小于 256，则 Resize 到 256 (保持长宽比)
            # 这种处理方式能确保 RandomCrop 永远不会报错
            new_w = max(self.least_size, w)
            new_h = max(self.least_size, h)
            if w < self.least_size:
                new_h = int(h * (self.least_size / w))
                new_w = self.least_size
            if h < self.least_size:
                new_w = int(w * (self.least_size / h))
                new_h = self.least_size
            A_img = A_img.resize((new_w, new_h), Image.BICUBIC)
            B_img = B_img.resize((new_w, new_h), Image.BICUBIC)

        # 1. 随机裁剪为 self.least_size * self.least_size
        i, j, h, w = transforms.RandomCrop.get_params(A_img, output_size=(self.least_size, self.least_size))
        A_img = F.crop(A_img, i, j, h, w)
        B_img = F.crop(B_img, i, j, h, w)

        # 2. 转换为 Tensor (0-1) 并归一化到 (-1, 1) 以匹配 train.py 逻辑
        A_tensor = self.transform(A_img)
        B_tensor = self.transform(B_img)

        # 3. 数据增强 (Flip)
        if not self.opt.no_flip:
            if random.random() < 0.5:
                A_tensor = F.hflip(A_tensor)
                B_tensor = F.hflip(B_tensor)
            if random.random() < 0.5:
                A_tensor = F.vflip(A_tensor)
                B_tensor = F.vflip(B_tensor)

        # 4. 亮度调整 (基于已经归一化到 -1~1 的 tensor)
        input_img = A_tensor
        if hasattr(self.opt, 'low_times') and random.random() < 0.5:
            times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
            # 这里的逻辑是将 -1~1 转回 0-1 调整亮度后再转回 -1~1
            input_img = ((A_tensor + 1.0) / 2.0 / (times + 1e-5)) * 2.0 - 1.0

        # 5. 计算灰度图
        r, g, b = (input_img[0:1] + 1), (input_img[1:2] + 1), (input_img[2:3] + 1)
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.

        return {
            'A': A_tensor,
            'B': B_tensor,
            'A_gray': A_gray, 
            'input_img': input_img,
            'A_paths': A_path, 
            'B_paths': B_path
        }

    def __len__(self):
        return self.A_size


# for ori BrighteningTrain dataset
# since the dataset is 384*384, we need not to crop or resize.
class PairDataset_2(data.Dataset):
    def __init__(self, opt):
        super(PairDataset_2, self).__init__()
        self.opt = opt

        # 直接在初始化中完成路径获取
        # self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        # self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        
        self.dir_A = os.path.join(opt.dataroot, "low")    # input low light
        self.dir_B = os.path.join(opt.dataroot, "high")   # gt

        self.A_paths = sorted(make_dataset(self.dir_A))
        self.B_paths = sorted(make_dataset(self.dir_B))
        self.A_size = len(self.A_paths)

        # 仅保留基础转换，动态增强放在 __getitem__
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index % self.A_size] # A 和 B 同名，一一对应的

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # 1. 基础转换
        A_img = self.transform(A_img)
        B_img = self.transform(B_img)

        # 2. 数据增强 (Flip)
        if not self.opt.no_flip:
            if random.random() < 0.5:
                A_img = F.hflip(A_img)
                B_img = F.hflip(B_img)
            if random.random() < 0.5:
                A_img = F.vflip(A_img)
                B_img = F.vflip(B_img)

        # 3. 亮度调整 (input_img 逻辑)
        input_img = A_img
        if hasattr(self.opt, 'low_times') and random.random() < 0.5:
            times = random.randint(self.opt.low_times, self.opt.high_times) / 100.
            # 将归一化后的数据转回 0-1 进行亮度模拟，再转回去
            input_img = ((A_img + 1.0) / 2.0 / (times + 1e-5)) * 2.0 - 1.0

        # 4. 计算灰度图 (用于平滑损失权重)
        # 使用标准的亮度公式，保持张量维度 [1, H, W]
        # 注意：这里的数据是归一化过的 (-1, 1)
        r, g, b = (input_img[0:1] + 1), (input_img[1:2] + 1), (input_img[2:3] + 1)
        A_gray = 1. - (0.299 * r + 0.587 * g + 0.114 * b) / 2.

        return {
            'A': A_img,     # input low light
            'B': B_img,     # gt
            'A_gray': A_gray, 
            'input_img': input_img,
            'A_paths': A_path, 
            'B_paths': B_path
        }

    def __len__(self):
        return self.A_size


class Config:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def Get_paired_dataset(batch_size):
    # 更新配置参数
    opt = Config(
        # dataroot="/home/mingmou/Downloads/retinexnet_datasets_ori/BrighteningTrain", 
        dataroot="/home/mingmou/Downloads/retinexnet_datasets_LSUI/LSUI_Split_Final/train", 
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