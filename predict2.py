# 作者: hxd
# 2023年04月23日16时33分47秒
import glob
import numpy as np
import torch
import os
import cv2
from torch import nn
from model import UNet
from data_loader import MyDataset
from torch.utils.data import DataLoader
from torchvision import transforms

def test():
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载网络，图片单通道，分类为1。
    net = UNet(3, 1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 加载模型参数
    net.load_state_dict(torch.load('best_model.pth', map_location=device))
    # 测试模式
    net.eval()
    # 加载训练数据集
    test_data = MyDataset("data/ImageSets/Segmentation", "data/JPEGImages", "data/SegmentationClass", transform,
                           "test")
    # 利用 DataLoader 来加载数据集
    test_dataloader = DataLoader(test_data, batch_size=1)
    criterion = nn.BCEWithLogitsLoss()
    # 遍历素有图片
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = net(imgs)
        loss = criterion(outputs, targets)
        print('测试集：Loss/test', loss.item())
        # # 提取结果
        # pred = np.array(pred.data.cpu()[0])[0]
        # # 处理结果
        # pred[pred >= 0.5] = 255
        # pred[pred < 0.5] = 0
        # # 保存图片
        # cv2.imwrite("data/res/"+str(i)+".bmp", pred)