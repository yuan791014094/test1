import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.tcunet import TCU_Net  # 导入TCU-Net模型
from src.utils.data_loader import RetinaDataset  # 导入数据加载器
from src.utils.metrics import dice_coefficient  # 导入评估指标

# 训练配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 10
BATCH_SIZE = 1
LEARNING_RATE = 0.0005
IMAGE_DIR = "data/train/images"
MASK_DIR = "data/train/annotations"
CHECKPOINT_DIR = "experiments/checkpoints"

# 加载数据
train_dataset = RetinaDataset(IMAGE_DIR, MASK_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# 初始化模型、损失函数和优化器
model = TCU_Net(in_channels=1, out_channels=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()  # 二值交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 学习率调度器
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for i, (images, masks) in enumerate(train_loader):
            images = images.float().unsqueeze(1).to(DEVICE)  # 单通道图像
            masks = masks.float().unsqueeze(1).to(DEVICE)    # 单通道标注

            # 检查目标标签的值
            # print("Target values:", masks)
            # assert torch.all((masks == 0) | (masks == 1)), "Target values must be 0 or 1"

            # 检查输入数据的范围
            # print("Input data range:", torch.min(images), torch.max(images))
            assert torch.all(images >= 0) and torch.all(images <= 1), "Input data must be normalized to [0, 1]"

            # 前向传播
            outputs = model(images)

            # 检查模型输出
            # print("Model output range:", torch.min(outputs), torch.max(outputs))
            loss = criterion(outputs, masks)
            print(f"Iter-{i} Loss: {loss.item():.4f}")

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # 更新学习率
        scheduler.step()

        # 打印训练信息
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

        # 保存模型权重
        # if (epoch + 1) % 10 == 0:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"model_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

# 开始训练
if __name__ == "__main__":
    train_model(model, train_loader, criterion, optimizer, scheduler, NUM_EPOCHS)