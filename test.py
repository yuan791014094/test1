import os
import torch
from torch.utils.data import DataLoader
from src.models.tcunet import TCU_Net  # 导入TCU-Net模型
from src.utils.data_loader import RetinaDataset  # 导入数据加载器
from src.utils.metrics import dice_coefficient, iou, accuracy, recall, f1_score  # 导入评估指标

# 测试配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
IMAGE_DIR = "data/test/images"
MASK_DIR = "data/test/annotations"
CHECKPOINT_PATH = "experiments/checkpoints/model_epoch_1.pth"

# 加载数据
test_dataset = RetinaDataset(IMAGE_DIR, MASK_DIR)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 初始化模型
model = TCU_Net(in_channels=1, out_channels=1).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.eval()

# 测试函数
def test_model(model, test_loader):
    model.eval()
    total_dice = 0.0
    total_iou = 0.0
    total_accuracy = 0.0
    total_recall = 0.0
    total_f1 = 0.0

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.unsqueeze(1).to(DEVICE).float()  # 单通道图像
            masks = masks.unsqueeze(1).to(DEVICE).float()    # 单通道标注

            # 前向传播
            outputs = model(images)
            outputs = torch.sigmoid(outputs)  # 将输出转换为概率
            outputs = (outputs > 0.5).float()  # 二值化

            # 计算评估指标
            dice = dice_coefficient(outputs, masks)
            iou_value = iou(outputs, masks)
            acc = accuracy(outputs, masks)
            rec = recall(outputs, masks)
            f1 = f1_score(outputs, masks)

            # 累加指标
            total_dice += dice.item()
            total_iou += iou_value.item()
            total_accuracy += acc.item()
            total_recall += rec.item()
            total_f1 += f1.item()

    # 计算平均指标
    avg_dice = total_dice / len(test_loader)
    avg_iou = total_iou / len(test_loader)
    avg_accuracy = total_accuracy / len(test_loader)
    avg_recall = total_recall / len(test_loader)
    avg_f1 = total_f1 / len(test_loader)

    # 打印测试结果
    print("------------------------------------")
    print(f"Average Dice Coefficient: {avg_dice:.4f}")
    print(f"Average IoU: {avg_iou:.4f}")
    print(f"Average Accuracy: {avg_accuracy:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print("------------------------------------")

# 开始测试
if __name__ == "__main__":
    test_model(model, test_loader)