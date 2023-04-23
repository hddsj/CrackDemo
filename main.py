# 作者: hxd
# 2023年04月18日17时11分48秒
import torch
from model import UNet
from torch import nn
from data_loader import MyDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import optim


def train_net(net,device,epochs=40,batch_size=1, lr=0.00001):
    transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
    #加载训练数据集
    train_data = MyDataset("data/ImageSets/Segmentation", "data/JPEGImages", "data/SegmentationClass", transform,
                           "train")
    # 利用 DataLoader 来加载数据集
    train_dataloader = DataLoader(train_data, batch_size=1)
    # 训练集的长度
    train_data_size = len(train_dataloader)
    # 定义RMSprop算法
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # 定义Loss算法
    criterion = nn.BCEWithLogitsLoss()
    # best_loss统计，初始化为正无穷
    best_loss = float('inf')
    #训练epochs次
    for epoch in range(epochs):
        #训练模式
        net.train()

        for data in train_dataloader:
            optimizer.zero_grad()
            imgs, targets = data
            #将数据拷贝到device中
            imgs = imgs.to(device)
            targets = targets.to(device)
            # 使用网络参数，输出预测结果
            outputs = net(imgs)
            # 计算loss
            loss = criterion(outputs, targets)
            print('第{}轮Loss/train{}'.format(epoch+1,loss))
            # 保存loss值最小的网络参数
            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            # 更新参数
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    # 定义设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 看是gpu还是cpu跑的
    print(torch.cuda.is_available())
    # 加载网络，图片单通道3，分类为2（因为是前景和背景，故填1）。
    net = UNet(3, 1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    #训练
    train_net(net, device)
