import torch
import os
import numpy as np
from PIL import Image
from torchvision import transforms
from models import Decom_Net, Enhance_Net

def get_test_transform():
    """预定义转换流程，避免循环内重复创建"""
    return transforms.Compose([
        transforms.ToTensor(),
        # 如果模型训练时用了 Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # 则测试也需要保持一致
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

@torch.no_grad() # 自动关闭梯度，提速减耗
def _test(model_path, save_dir, test_data_dir, device, num_layer):
    # 1. 初始化并加载模型
    dec = Decom_Net(num_layer).to(device)
    enh = Enhance_Net().to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    # 注意：如果训练时用了 DataParallel，保存的 key 会带 'module.' 前缀
    # 这里需要特殊处理或确保保存时用的是 model.module.state_dict()
    dec_state = {k.replace('module.', ''): v for k, v in checkpoint["Dec_model"].items()}
    enh_state = {k.replace('module.', ''): v for k, v in checkpoint["Enh_model"].items()}
    
    dec.load_state_dict(dec_state)
    enh.load_state_dict(enh_state)
    dec.eval() # 切换到评价模式
    enh.eval()

    test_transform = get_test_transform()
    os.makedirs(save_dir, exist_ok=True)

    # 2. 遍历测试
    img_names = [f for f in os.listdir(test_data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_name in img_names:
        full_path = os.path.join(test_data_dir, img_name)
        img_raw = Image.open(full_path).convert('RGB')
        
        # 预处理
        img_tensor = test_transform(img_raw).unsqueeze(0).to(device)
        # 将 (-1, 1) 映射到 (0, 1) 以符合 Retinex 理论输入
        img_input = (img_tensor + 1.0) / 2.0

        # 推理
        R, I = dec(img_input)
        I_hat = enh(R, I)
        
        # 合成增强后的图像 (S = R * I)
        # 注意：如果 Enhance_Net 输出的是 I_hat 增强系数，则 S_hat = R * I_hat
        S_hat = R * I_hat
        
        # 3. 后处理并保存
        # Clamp 保证像素值在 [0, 1] 之间
        S_hat = torch.clamp(S_hat, 0, 1)
        result_img = transforms.ToPILImage()(S_hat.squeeze(0).cpu())
        
        result_img.save(os.path.join(save_dir, img_name))
        print(f"Processed: {img_name}")

if __name__ == "__main__":
    # 配置
    CONFIG = {
        "num_layer": 5,
        "model_path": "./checkpoints/model_200.tar",
        "test_data_dir": "/home/mingmou/Downloads/retinexnet_datasets/test_dataset/testA",
        "save_dir": "./test_dataset/resultsA/",
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu")
    }

    _test(
        CONFIG["model_path"], 
        CONFIG["save_dir"], 
        CONFIG["test_data_dir"], 
        CONFIG["device"],
        CONFIG["num_layer"]
    )