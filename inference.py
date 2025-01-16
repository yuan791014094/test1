import os
import cv2
import torch
import numpy as np
from src.models.tcunet import TCU_Net  # 导入TCU-Net模型

# 推理配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "experiments/checkpoints/model_epoch_1.pth"
INPUT_IMAGE_DIR = "data/test/images"  # 输入图像文件夹
OUTPUT_MASK_DIR = "experiments/results"  # 输出分割结果文件夹

# 加载模型
model = TCU_Net(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# 图像预处理函数
def preprocess_image(image):
    """
    对输入图像进行预处理
    :param image: 输入的RGB图像（H, W, 3）
    :return: 预处理后的单通道图像（1, H, W）
    """
    # 1. 提取G通道并灰度化
    gray_image = image[:, :, 1]  # 提取G通道

    # 2. 归一化
    gray_image = gray_image / 255.0  # 归一化到[0, 1]范围

    # 3. 对比度受限的直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(np.uint8(gray_image * 255))  # 转换为uint8类型
    clahe_image = clahe_image / 255.0  # 归一化到[0, 1]范围

    # 4. 伽马校正（非线性化增亮）
    gamma = 1.5  # 伽马值
    gamma_corrected = np.power(clahe_image, 1 / gamma)  # 伽马校正

    # 转换为PyTorch张量
    tensor_image = torch.tensor(gamma_corrected, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    return tensor_image

# 推理函数
def run_inference(image_path, output_path):
    """
    对单张图像进行推理
    :param image_path: 输入图像路径
    :param output_path: 输出分割结果路径
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # 预处理图像
    tensor_image = preprocess_image(image).to(DEVICE)

    # 模型推理
    with torch.no_grad():
        output = model(tensor_image)
        output = torch.sigmoid(output)  # 将输出转换为概率
        output = (output > 0.5).float()  # 二值化

    # 后处理：将输出转换为NumPy数组
    mask = output.squeeze(0).squeeze(0).cpu().numpy()  # (H, W)

    # 保存分割结果
    mask_uint8 = (mask * 255).astype(np.uint8)  # 转换为0-255范围
    cv2.imwrite(output_path, mask_uint8)  # 保存分割结果
    print(f"Saved segmentation result to {output_path}")

# 批量推理
def batch_inference(input_dir, output_dir):
    """
    对输入文件夹中的所有图像进行推理，并保存分割结果
    :param input_dir: 输入图像文件夹
    :param output_dir: 输出分割结果文件夹
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            run_inference(image_path, output_path)

# 主函数
if __name__ == "__main__":
    batch_inference(INPUT_IMAGE_DIR, OUTPUT_MASK_DIR)