# 作者: hxd
# 2023年04月18日17时12分55秒
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

class MyDataset(Dataset):
    def __init__(self,txt_dir,image_dir,label_dir,transform,type):
        self.txt_dir = txt_dir
        self.img_dir =image_dir
        self.label_dir = label_dir
        self.transform = transform
        global file
        if type == "train":
            file = open(os.path.join(txt_dir,"SubFCNDemotrain.txt"))
        elif type == 'test':
            file = open(os.path.join(txt_dir,"SubFCNDemotest.txt"))
        elif type == 'val':
            file = open(os.path.join(txt_dir,"FCNDemoval.txt"))
        self.file_list = file.read().splitlines()


    def __getitem__(self, item):
        img_path = os.path.join(self.img_dir,self.file_list[item])+".jpg"
        label_path = os.path.join(self.label_dir,self.file_list[item])+".bmp"
        img = Image.open(img_path)


        label = Image.open(label_path)
        img = self.transform(img)
        label = self.transform(label)
        return (img,label)

    def __len__(self):
        return len(self.file_list)

if __name__ == '__main__':
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    train_data = MyDataset("data/ImageSets/Segmentation", "data/JPEGImages", "data/SegmentationClass", transform, "train")
    test_data = MyDataset("data/ImageSets/Segmentation", "data/JPEGImages", "data/SegmentationClass", transform, "test")
    # length 长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(train_data_size)
    print(test_data_size)
    # 利用 DataLoader 来加载数据集
    train_dataloader = DataLoader(train_data, batch_size=1)
    test_dataloader = DataLoader(test_data, batch_size=1)
    for data in train_dataloader:
        img, target = data
        print("img:"+str(img.shape))
        print("target:"+str(target.shape))