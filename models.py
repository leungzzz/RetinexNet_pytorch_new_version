import torch
import torch.nn as nn
import torch.nn.functional as F
class Decom_Net(nn.Module):
    def __init__(self, num_layers=5):
        super(Decom_Net, self).__init__()
        # 初始输入层：处理拼接后的 4 通道 (RGB + Max_Intensity)
        self.head = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True)
        )
        # 中间特征提取层
        mid_layers = []
        for _ in range(num_layers):
            mid_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
            mid_layers.append(nn.ReLU(inplace=True))
        self.body = nn.Sequential(*mid_layers)
        # 输出层：输出 4 通道 (3 为反射率 R, 1 为照度 I)
        self.tail = nn.Sequential(
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        # 预处理：拼接最大亮度通道
        mx = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat((mx, x), dim=1)
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        R = out[:, 0:3, :, :]
        I = out[:, 3:4, :, :]
        return R, I
class Enhance_Net(nn.Module):
    def __init__(self):
        super(Enhance_Net, self).__init__()
        # Encoder (下采样)
        self.conv1 = nn.Sequential(nn.Conv2d(4, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True))
        # Decoder (上采样)
        self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True))
        # Multi-scale Fusion
        # 将不同尺度的特征融合，这里我们显式定义融合层
        self.fusion = nn.Sequential(
            nn.Conv2d(64*3 + 64, 64, kernel_size=1), # 64*3 是跳跃连接，+64 是上采样的特征
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # 照度增强通常输出 0-1 的增量或图层
        )
    def forward(self, R, I):
        x_in = torch.cat((R, I), dim=1)
        # Encoder
        h1 = self.conv1(x_in)  # 1/2
        h2 = self.conv2(h1)    # 1/4
        h3 = self.conv3(h2)    # 1/8
        # Decoder with Skip Connections
        up1 = self.up1(h3)
        up2 = self.up2(torch.cat((up1, h2), dim=1))
        up3 = self.up3(torch.cat((up2, h1), dim=1))
        # Multi-scale feature alignment (使用 F.interpolate 代替临时实例化)
        c1 = F.interpolate(h1, size=x_in.shape[2:], mode='nearest')
        c2 = F.interpolate(h2, size=x_in.shape[2:], mode='nearest')
        c3 = F.interpolate(h3, size=x_in.shape[2:], mode='nearest')
        # 最终融合
        fused = torch.cat([up3, c1, c2, c3], dim=1)
        # 注意：这里需要根据 fusion 层的 input channel 调整 cat 的对象
        # 原代码逻辑比较奇特，这里建议精简为只对解码器输出做处理，或者修正通道数
        # 修正原代码中 321 通道的奇怪逻辑，建议只使用 up3 结果
        # 如果坚持原逻辑，请确保 fusion 层 in_channels = up3.shape[1] + c1...c3.shape[1]
        out = self.fusion(fused)
        return out
    

####################################################################################

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class Decom_Net(nn.Module):
#     def __init__(self, num_layers=5):
#         super(Decom_Net, self).__init__()
#         # 初始输入层：处理拼接后的 4 通道 (RGB + Max_Intensity)
#         self.head = nn.Sequential(
#             nn.Conv2d(4, 64, kernel_size=9, stride=1, padding=4),
#             nn.ReLU(inplace=True)
#         )
        
#         # 中间特征提取层
#         mid_layers = []
#         for _ in range(num_layers):
#             mid_layers.append(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1))
#             mid_layers.append(nn.ReLU(inplace=True))
#         self.body = nn.Sequential(*mid_layers)
        
#         # 输出层：输出 4 通道 (3 为反射率 R, 1 为照度 I)
#         self.tail = nn.Sequential(
#             nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
#             nn.Sigmoid()
#         )

#     def forward(self, x):
#         # 预处理：拼接最大亮度通道
#         mx = torch.max(x, dim=1, keepdim=True)[0]
#         x = torch.cat((mx, x), dim=1)
        
#         feat = self.head(x)
#         feat = self.body(feat)
#         out = self.tail(feat)
        
#         R = out[:, 0:3, :, :]
#         I = out[:, 3:4, :, :]
#         return R, I

# class Enhance_Net(nn.Module):
#     def __init__(self):
#         super(Enhance_Net, self).__init__()
#         # Encoder (下采样)
#         self.conv1 = nn.Sequential(nn.Conv2d(4, 64, 4, 2, 1), nn.ReLU(inplace=True))
#         self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True))

#         # Decoder (上采样)
#         self.up1 = nn.Sequential(nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(inplace=True))
#         self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True))
#         self.up3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(inplace=True))

#         # Multi-scale Fusion
#         # 将不同尺度的特征融合，这里我们显式定义融合层
#         self.fusion = nn.Sequential(
#             nn.Conv2d(64*3 + 64, 64, kernel_size=1), # 64*3 是跳跃连接，+64 是上采样的特征
#             nn.ReLU(inplace=True),
#             nn.Conv2d(64, 1, kernel_size=3, padding=1),
#             nn.Sigmoid() # 照度增强通常输出 0-1 的增量或图层
#         )

#     def forward(self, R, I):
#         x_in = torch.cat((R, I), dim=1)
        
#         # Encoder
#         h1 = self.conv1(x_in)  # 1/2
#         h2 = self.conv2(h1)    # 1/4
#         h3 = self.conv3(h2)    # 1/8
        
#         # Decoder with Skip Connections
#         up1 = self.up1(h3)
#         up2 = self.up2(torch.cat((up1, h2), dim=1))
#         up3 = self.up3(torch.cat((up2, h1), dim=1))
        
#         # Multi-scale feature alignment (使用 F.interpolate 代替临时实例化)
#         c1 = F.interpolate(h1, size=x_in.shape[2:], mode='nearest')
#         c2 = F.interpolate(h2, size=x_in.shape[2:], mode='nearest')
#         c3 = F.interpolate(h3, size=x_in.shape[2:], mode='nearest')
        
#         # 最终融合
#         fused = torch.cat([up3, c1, c2, c3], dim=1)
#         # 注意：这里需要根据 fusion 层的 input channel 调整 cat 的对象
#         # 原代码逻辑比较奇特，这里建议精简为只对解码器输出做处理，或者修正通道数
        
#         # 修正原代码中 321 通道的奇怪逻辑，建议只使用 up3 结果
#         # 如果坚持原逻辑，请确保 fusion 层 in_channels = up3.shape[1] + c1...c3.shape[1]
#         out = self.fusion(fused) 
#         return out

