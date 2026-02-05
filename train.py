import torch
import torch.nn as nn
import torch.optim as optim
import os
from models import Decom_Net, Enhance_Net
from torch.utils.data import DataLoader
from datapipline import Get_paired_dataset

import torchvision.utils as vutils # 需要导入这个工具包用于保存图片

# --------------------------------------- 辅助模块 ---------------------------------------
class GradientLoss(nn.Module):
    """将原本低效的 get_smooth 封装为 Module，预存权重以提高性能"""
    def __init__(self):
        super(GradientLoss, self).__init__()
        # 定义 x, y 方向的卷积核
        kernel_x = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).view(1, 1, 3, 3)
        kernel_y = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).view(1, 1, 3, 3)
        self.register_buffer('weight_x', kernel_x)
        self.register_buffer('weight_y', kernel_y)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, I, R):
        R_gray = torch.mean(R, dim=1, keepdim=True)
        # 计算梯度
        grad_I_x = torch.abs(torch.nn.functional.conv2d(I, self.weight_x, padding=1))
        grad_I_y = torch.abs(torch.nn.functional.conv2d(I, self.weight_y, padding=1))
        # 计算结构感知权重
        grad_R_x = self.avg_pool(torch.abs(torch.nn.functional.conv2d(R_gray, self.weight_x, padding=1)))
        grad_R_y = self.avg_pool(torch.abs(torch.nn.functional.conv2d(R_gray, self.weight_y, padding=1)))
        
        loss = torch.mean(grad_I_x * torch.exp(-10 * grad_R_x) + grad_I_y * torch.exp(-10 * grad_R_y))
        return loss

# --------------------------------------- 核心逻辑 ---------------------------------------
def train():
    # 参数设置
    epochs = 100   # ori 200
    lr = 1e-4
    nums_layer = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 模型初始化
    dec_model = Decom_Net(nums_layer).to(device)
    enh_model = Enhance_Net().to(device)
    
    if torch.cuda.device_count() > 1:
        dec_model = nn.DataParallel(dec_model)
        enh_model = nn.DataParallel(enh_model)

    # 2. 优化器
    opt_dec = optim.Adam(dec_model.parameters(), lr=lr)
    opt_enh = optim.Adam(enh_model.parameters(), lr=lr)
    
    # 3. 损失函数
    l1_loss = nn.L1Loss()
    grad_loss_func = GradientLoss().to(device)

    # 4. 数据准备 (假设 datapipline 返回的是标准 Dataset)
    # dataset = Get_paired_dataset(1) 
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    dataloader = Get_paired_dataset(batch_size=1) # 直接获取 dataloader

    save_dir = "./training_results"
    os.makedirs(save_dir, exist_ok=True)

    # 5. 训练循环
    for epoch in range(epochs):
        dec_model.train()
        enh_model.train()

        # 在每个 Epoch 结束时，或者每隔固定步数保存一次可视化结果
        if (epoch + 1) % 10 == 0:
            dec_model.eval()
            enh_model.eval()
            with torch.no_grad():
                # 取当前 batch 的数据进行可视化
                # s_low 是输入的低光图
                # r_low, i_low 是分解出的结果
                # i_low_hat 是增强后的照度
                # s_hat = r_low * i_low_hat 是最终增强图
                s_hat = r_low_fixed * i_low_hat
                
                # 将照度图 I 扩充为 3 通道以便与 R 拼接显示
                i_low_3ch = torch.cat([i_low_fixed, i_low_fixed, i_low_fixed], dim=1)
                i_hat_3ch = torch.cat([i_low_hat, i_low_hat, i_low_hat], dim=1)

                # 拼接：第一行 [输入图, 分解出的R, 分解出的I]
                # 第二行 [真值图S_normal, 增强后的I_hat, 最终结果S_hat]
                image_grid = torch.cat([
                    s_low, r_low_fixed, i_low_3ch,
                    s_normal, i_hat_3ch, s_hat
                ], dim=0)

                # 保存为网格图
                vutils.save_image(
                    image_grid, 
                    f"{save_dir}/epoch_{epoch+1}.png", 
                    nrow=3, 
                    normalize=False # 因为我们的数据已经在 [0, 1] 之间
                )
            dec_model.train()
            enh_model.train()
        
        epoch_dec_loss = 0
        epoch_enh_loss = 0

        for i, data in enumerate(dataloader):
            # 改动点：直接搬运数据，不再执行 ((x + 1) / 2)
            s_low = data['A'].to(device)
            s_normal = data['B'].to(device)

            # --- 阶段 A: 训练分解网络 (Decom_Net) ---
            opt_dec.zero_grad()
            
            r_low, i_low = dec_model(s_low)
            r_normal, i_normal = dec_model(s_normal)

            # 重构损失 (S = R * I)
            loss_reconst = l1_loss(s_low, r_low * i_low) + \
                           l1_loss(s_normal, r_normal * i_normal) + \
                           0.001 * l1_loss(s_low, r_normal * i_low) + \
                           0.001 * l1_loss(s_normal, r_low * i_normal)
            
            # 反射率一致性 (Retinex 核心：不同光照下反射率应相同)
            loss_ivref = 0.1 * l1_loss(r_low, r_normal) # 建议提高权重到 0.1 以抑制噪声
            
            # 平滑损失 (利用 A_gray 或 R 引导 I 的平滑)
            loss_smooth_dec = 0.1 * grad_loss_func(i_low, r_low) + \
                              0.1 * grad_loss_func(i_normal, r_normal)
            
            loss_dec = loss_reconst + loss_ivref + loss_smooth_dec
            loss_dec.backward()
            opt_dec.step()
            
            # --- 阶段 B: 训练增强网络 (Enhance_Net) ---
            opt_enh.zero_grad()
            
            with torch.no_grad():
                # 训练增强网络时，Decom_Net 的参数固定
                r_low_fixed, i_low_fixed = dec_model(s_low)
            
            i_low_hat = enh_model(r_low_fixed.detach(), i_low_fixed.detach())
            
            # 增强后的图像应与正常光照图像 S_normal 接近
            loss_reconst_enh = l1_loss(s_normal, r_low_fixed.detach() * i_low_hat)
            loss_smooth_enh = 3.0 * grad_loss_func(i_low_hat, r_low_fixed.detach())
            
            loss_enh = loss_reconst_enh + loss_smooth_enh
            loss_enh.backward()
            opt_enh.step()

        # 打印日志
        print(f"Epoch [{epoch+1}/{epochs}] | Dec Loss: {epoch_dec_loss/len(dataloader):.6f} | Enh Loss: {epoch_enh_loss/len(dataloader):.6f}")

        # 6. 定期保存
        if (epoch + 1) % 10 == 0:
            save_path = "./checkpoints_2/"
            os.makedirs(save_path, exist_ok=True)
            checkpoint = {
                "Dec_model": dec_model.state_dict(),
                "Enh_model": enh_model.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint, f"{save_path}model_{epoch+1}.tar")
            torch.save(checkpoint, f"{save_path}model_latest.tar")

if __name__ == '__main__':
    train()