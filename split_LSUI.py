import os
import random
import shutil
from pathlib import Path

def split_lsui_dataset(src_input_dir, src_gt_dir, output_root, train_ratio=0.9):
    """
    针对 LSUI 数据集的随机切分脚本
    """
    # 转换为 Path 对象，方便处理 Linux 路径
    src_input_dir = Path(src_input_dir)
    src_gt_dir = Path(src_gt_dir)
    output_root = Path(output_root)

    # 设置随机种子，确保实验可重复
    random.seed(42)

    # 获取所有图片文件名
    # LSUI 的 input 和 GT 文件名通常是一一对应的
    image_list = [f.name for f in src_input_dir.glob("*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    
    if not image_list:
        print(f"错误: 在 {src_input_dir} 中没找到图片，请检查路径。")
        return

    random.shuffle(image_list)

    # 计算切分界限
    split_idx = int(len(image_list) * train_ratio)
    train_list = image_list[:split_idx]
    test_list = image_list[split_idx:]

    # 创建符合 RetinexNet 项目要求的目录结构
    for mode in ['train', 'test']:
        (output_root / mode / 'low').mkdir(parents=True, exist_ok=True)
        (output_root / mode / 'high').mkdir(parents=True, exist_ok=True)

    def move_and_copy(files, mode):
        print(f"正在处理 {mode} 集合 (共 {len(files)} 张)...")
        for filename in files:
            # 拷贝 input 图像到 low 文件夹
            shutil.copy2(src_input_dir / filename, output_root / mode / 'low' / filename)
            # 拷贝 GT 图像到 high 文件夹
            shutil.copy2(src_gt_dir / filename, output_root / mode / 'high' / filename)

    move_and_copy(train_list, 'train')
    move_and_copy(test_list, 'test')
    
    print(f"\n 数据集切分完成！")
    print(f"总计: {len(image_list)} 对")
    print(f"训练集 (90%): {len(train_list)} 对 -> {output_root}/train")
    print(f"测试集 (10%): {len(test_list)} 对 -> {output_root}/test")

if __name__ == "__main__":
    # 根据你的目录结构配置路径
    SOURCE_INPUT = "/home/mingmou/Downloads/retinexnet_datasets_LSUI/backup/input"
    SOURCE_GT = "/home/mingmou/Downloads/retinexnet_datasets_LSUI/backup/GT"
    
    # 建议将输出结果放在 Downloads 下的一个新文件夹里
    TARGET_DATASET = "/home/mingmou/Downloads/retinexnet_datasets_LSUI/LSUI_Split_Final"
    
    split_lsui_dataset(SOURCE_INPUT, SOURCE_GT, TARGET_DATASET, train_ratio=0.9)