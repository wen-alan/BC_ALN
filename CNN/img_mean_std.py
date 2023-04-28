# from torchvision.transforms import ToTensor #用于把图片转化为张量
import transforms as transforms
from BC_data import MyDataset
import numpy as np #用于将张量转化为数组，进行除法
# from torchvision.datasets import ImageFolder #用于导入图片数据集

Dataroot = 'data/BC650_jpg_scale'
# Dataroot = 'data/seg_val_jpg'
# Dataroot = 'data/seg_val2_jpg'
# Dataroot = 'data/seg_val3_jpg'
#初始化均值和方差
means = 0.0
stds = 0.0
#可将图片类型转化为张量，并把0~255的像素值缩小到0~1之间
transform_test = transforms.Compose([
    transforms.Resize((70, 70)),
    transforms.CenterCrop(64),
    transforms.ToTensor()])
#导入数据集的图片，并且转化为张量
dataset = MyDataset(split = 'train', data_dir=Dataroot, transform=transform_test, ratio=1.0)
num_imgs=len(dataset)
#遍历数据集的张量和标签
for img, lab in dataset:
    # 计算每一个通道的均值和标准差
    means += img[ :, :].mean()
    stds += img[ :, :].std()
#要使数据集归一化，均值和方差需除以总图片数量
mean = means/num_imgs
std = stds/num_imgs
print("mean, std:",mean, std)

