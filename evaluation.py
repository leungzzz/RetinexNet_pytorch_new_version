import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

class Evaluator:
    def __init__(self, gt_dir, result_dir):
        self.gt_dir = gt_dir
        self.result_dir = result_dir
        # 自动匹配同名文件（更稳健）
        self.file_list = [f for f in os.listdir(result_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
    def calculate_scores(self):
        psnr_list = []
        ssim_list = []

        for file_name in self.file_list:
            # 1. 增强图的路径（当前遍历到的 lowxxxxx.png）
            res_path = os.path.join(self.result_dir, file_name)
            
            # 2. 生成对应的真值图文件名（将 low 替换为 normal）
            # 例如：low00001.png -> normal00001.png
            gt_file_name = file_name.replace("low", "normal")
            gt_path = os.path.join(self.gt_dir, gt_file_name)

            # 检查文件是否存在
            if not os.path.exists(gt_path):
                print(f"Warning: Ground Truth not found for {gt_file_name} (from {file_name}), skipping.")
                continue

            # 读取并计算
            try:
                img_res = np.array(Image.open(res_path).convert('RGB'))
                img_gt = np.array(Image.open(gt_path).convert('RGB'))

                # 如果尺寸不一致，强制缩放到一致（防止报错）
                if img_res.shape != img_gt.shape:
                    img_gt = np.array(Image.fromarray(img_gt).resize((img_res.shape[1], img_res.shape[0]), Image.BICUBIC))

                cur_psnr = psnr_func(img_gt, img_res, data_range=255)
                cur_ssim = ssim_func(img_gt, img_res, data_range=255, channel_axis=2)

                psnr_list.append(cur_psnr)
                ssim_list.append(cur_ssim)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

        return psnr_list, ssim_list

if __name__ == "__main__":
    # 修改实际路径
    # gt_dir 是 Ground Truth (正常光照图)，result_dir 是模型生成图路径
    gt_dir = "/home/mingmou/Downloads/retinexnet_datasets/test_dataset/testB/"      # 参考图
    result_dir = "./test_dataset/resultsA/" # 增强图
    
    evaluator = Evaluator(gt_dir, result_dir)
    psnrs, ssims = evaluator.calculate_scores()
    
    if psnrs:
        print("-" * 30)
        print(f"Evaluation Results:")
        print(f"Average PSNR: {np.mean(psnrs):.4f} dB")
        print(f"Average SSIM: {np.mean(ssims):.4f}")
        print("-" * 30)
    else:
        print("No paired images found for evaluation.")