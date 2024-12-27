import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

#自定义激活函数部分：
#Hswish 和 Hsigmoid 函数：
#定义了 Hswish 和 Hsigmoid 这两个自定义的激活函数，它们基于 torch 中的函数（如 F.relu6）进行了特定的数学运算组合，用于在网络
# 结构中引入新的非线性变换方式，符合 MobileNetV3 网络对激活函数的特定设计要求。
def Hswish(x,inplace=True):
    return x * F.relu6(x + 3., inplace=inplace) / 6.

def Hsigmoid(x,inplace=True):
    return F.relu6(x + 3., inplace=inplace) / 6.

#SEModule 类（Squeeze-And-Excite 模块）：
#结构和功能：
#这是一个实现了通道注意力机制的模块，通过自适应平均池化（nn.AdaptiveAvgPool2d）将输入特征图在空间维度上压缩为 1x1，然后经过两个
# 全连接层（nn.Linear）以及自定义的 Hsigmoid 激活函数，得到每个通道的权重，最后将这个权重与原始输入特征图在通道维度上进行相乘，
# 实现对不同通道重要性的重新加权，有助于网络更好地聚焦于关键特征信息，提升模型性能。
# Squeeze-And-Excite模块
class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y=self.avg_pool(x).view(b, c)
        y=self.se(y)
        y = Hsigmoid(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

#Bottleneck 类（瓶颈结构模块）：
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,exp_channels,stride,se='True',nl='HS'):
        # 初始化部分：
        # 在 __init__ 方法中，根据传入的参数（如输入输出通道数、卷积核大小、是否使用 SEModule、激活函数类型等）初始化了一系列的卷积层
        # （nn.Conv2d）、批归一化层（nn.BatchNorm2d）以及确定了激活函数（通过 self.nlin_layer 根据传入的 nl 参数选择），同时根据步长
        # 和输入输出通道情况初始化了残差连接（self.shortcut），构建起了一个完整的瓶颈结构，该结构是 MobileNetV3 网络中的基本构建单元，
        # 融合了深度可分离卷积、通道注意力机制以及残差连接等技术，用于高效地提取特征并控制模型复杂度。
        super(Bottleneck, self).__init__()
        padding = (kernel_size - 1) // 2
        if nl == 'RE':
            self.nlin_layer = F.relu6
        elif nl == 'HS':
            self.nlin_layer = Hswish
        self.stride=stride
        if se:
            self.se=SEModule(exp_channels)
        else:
            self.se=None
        self.conv1=nn.Conv2d(in_channels,exp_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn1 = nn.BatchNorm2d(exp_channels)
        self.conv2=nn.Conv2d(exp_channels,exp_channels,kernel_size=kernel_size,stride=stride,
                             padding=padding,groups=exp_channels,bias=False)
        self.bn2=nn.BatchNorm2d(exp_channels)
        self.conv3=nn.Conv2d(exp_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn3=nn.BatchNorm2d(out_channels)
        # 先初始化一个空序列，之后改造其成为残差链接
        self.shortcut = nn.Sequential()
        # 只有步长为1且输入输出通道不相同时才采用跳跃连接(想一下跳跃链接的过程，输入输出通道相同这个跳跃连接就没意义了)
        if stride == 1 and in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 下面的操作卷积不改变尺寸，仅匹配通道数
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    #forward 方法：
    #在 forward 方法中，按照设计好的顺序依次对输入 x 进行卷积、归一化、激活等操作，根据是否启用 SEModule 进行相应的处理，并在
    # 最后根据步长情况决定是否添加残差连接，实现了特征的前向传播计算，保证了数据在这个瓶颈结构中的正确流动和特征变换。
    def forward(self,x):
        out=self.nlin_layer(self.bn1(self.conv1(x)))
        if self.se is not None:
            out=self.bn2(self.conv2(out))
            out=self.nlin_layer(self.se(out))
        else:
            out = self.nlin_layer(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride == 1 else out
        return out

#MobileNetV3_large 和 MobileNetV3_small 类（网络主体类）：
class MobileNetV3_large(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg=[
        (16,3,16,1,False,'RE'),
        (24,3,64,2,False,'RE'),
        (24,3,72,1,False,'RE'),
        (40,5,72,2,True,'RE'),
        (40,5,120,1,True,'RE'),
        (40,5,120,1,True,'RE'),
        (80,3,240,2,False,'HS'),
        (80,3,200,1,False,'HS'),
        (80,3,184,1,False,'HS'),
        (80,3,184,1,False,'HS'),
        (112,3,480,1,True,'HS'),
        (112,3,672,1,True,'HS'),
        (160,5,672,2,True,'HS'),
        (160,5,960,1,True,'HS'),
        (160,5,960,1,True,'HS')
    ]
    #网络结构搭建：
    #在各自的 __init__ 方法中，首先初始化了网络的第一层卷积层和批归一化层，然后通过调用 _make_layers 方法根据预定义的配置参数
    # cfg 列表自动生成一系列的 Bottleneck 层构建网络的主体结构，接着定义了后续用于进一步特征变换和分类输出的卷积层（如 conv2、
    # conv3、conv4 等），通过这样的层次结构搭建起了完整的 MobileNetV3 网络（大版本和小版本在层数、通道数等配置上有所不同，以适
    # 应不同的计算资源和性能需求）。
    def __init__(self,num_classes=17):
        super(MobileNetV3_large,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        self.conv2=nn.Conv2d(160,960,1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(960)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3=nn.Conv2d(960,1280,1,1,padding=0,bias=True)
        self.conv4=nn.Conv2d(1280,num_classes,1,stride=1,padding=0,bias=True)

    def _make_layers(self,in_channels):
        layers=[]
        for out_channels,kernel_size,exp_channels,stride,se,nl in self.cfg:
            layers.append(
                Bottleneck(in_channels,out_channels,kernel_size,exp_channels,stride,se,nl)
            )
            in_channels=out_channels
        return nn.Sequential(*layers)
    #forward 方法：
    #在 forward 方法中，定义了数据从输入开始经过各个层的完整前向传播路径，包括激活函数的应用、特征池化、维度调整等操作，最终
    # 输出符合分类任务要求的预测结果，并且注意到在最后对输出结果进行了 view 操作，将四维的卷积层输出结果（最后两维为 1）转换
    # 为二维形式，以适配后续可能的损失函数计算等操作要求，整体符合深度学习网络进行前向推理的逻辑流程。
    def forward(self,x):
        out=Hswish(self.bn1(self.conv1(x)))
        out=self.layers(out)
        out=Hswish(self.bn2(self.conv2(out)))
        out=F.avg_pool2d(out,7)
        out=Hswish(self.conv3(out))
        out=self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需
        # 要做一个resize
        a,b=out.size(0),out.size(1)
        out=out.view(a,b)
        return out

class MobileNetV3_small(nn.Module):
    # (out_channels,kernel_size,exp_channels,stride,se,nl)
    cfg = [
        (16,3,16,2,True,'RE'),
        (24,3,72,2,False,'RE'),
        (24,3,88,1,False,'RE'),
        (40,5,96,2,True,'HS'),
        (40,5,240,1,True,'HS'),
        (40,5,240,1,True,'HS'),
        (48,5,120,1,True,'HS'),
        (48,5,144,1,True,'HS'),
        (96,5,288,2,True,'HS'),
        (96,5,576,1,True,'HS'),
        (96,5,576,1,True,'HS')
    ]
    def __init__(self,num_classes=17):
        super(MobileNetV3_small,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,2,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(16)
        # 根据cfg数组自动生成所有的Bottleneck层
        self.layers = self._make_layers(in_channels=16)
        self.conv2=nn.Conv2d(96,576,1,stride=1,bias=False)
        self.bn2=nn.BatchNorm2d(576)
        # 卷积后不跟BN，就应该把bias设置为True
        self.conv3=nn.Conv2d(576,1280,1,1,padding=0,bias=True)
        self.conv4=nn.Conv2d(1280,num_classes,1,stride=1,padding=0,bias=True)

    def _make_layers(self,in_channels):
        layers=[]
        for out_channels,kernel_size,exp_channels,stride,se,nl in self.cfg:
            layers.append(
                Bottleneck(in_channels,out_channels,kernel_size,exp_channels,stride,se,nl)
            )
            in_channels=out_channels
        return nn.Sequential(*layers)

    def forward(self,x):
        out=Hswish(self.bn1(self.conv1(x)))
        out=self.layers(out)
        out=self.bn2(self.conv2(out))
        se=SEModule(out.size(1))
        out=Hswish(se(out))
        out = F.avg_pool2d(out, 7)
        out = Hswish(self.conv3(out))
        out = self.conv4(out)
        # 因为原论文中最后一层是卷积层来实现全连接的效果，维度是四维的，后两维是1，在计算损失函数的时候要求二维，因此在这里需要做一个resize
        a, b = out.size(0), out.size(1)
        out = out.view(a, b)
        return out
#数据加载逻辑：
#这个类继承自 torch.utils.data.Dataset，用于加载图像数据。在 __init__ 方法中接收图像路径列表 img_paths 和图像变换操作
# transform，在 __getitem__ 方法中通过 PIL 库的 Image.open 打开图像，并应用传入的变换操作进行预处理，最后返回处理后的图像数据，
# __len__ 方法则返回图像路径列表的长度，方便后续配合 DataLoader 使用，实现了一个简单但实用的图像数据集加载机制，符合 PyTorch
# 中数据加载的规范模式。
class ImageDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_paths)

# 测试代码，跑通证明网络结构没问题
#通过创建一个随机输入张量（模拟输入图像数据的批次、通道、高度和宽度维度），传入 MobileNetV3_small 模型进行前向传播，然后打印
# 输出结果的尺寸和内容，用于简单验证网络结构是否能够正确运行，是否存在明显的语法错误或者维度不匹配等导致运行失败的问题，在模型
# 开发和调试阶段起到了初步的测试作用。
def test():
    net=MobileNetV3_small()
    x=torch.randn(2,3,224,224)
    y=net(x)
    print(y.size())
    print(y)

if __name__ == '__main__':
    model = MobileNetV3_large()
    #model = MobileNetV3_small()
    img_path = 'elephant.jpg'
    img = Image.open(img_path).resize((224, 224))  # PIL中使用Image.open打开图像，并可直接调用resize方法调整大小
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.transpose(x, (0, 3, 1, 2))  # 将维度顺序调整为(batch_size, height, width, channels)
    #x = x.detach().numpy()
    x = preprocess_input(x)
    print('Input image shape:', x.shape)
    x = x.copy()  # 添加这行代码，创建一个副本，解决可能的负步长问题
    with torch.no_grad():  # 一般在预测时关闭梯度计算，节省内存并提高效率
        x = torch.from_numpy(x).float() # 将numpy数组转换为torch张量，并设置数据类型为float
        preds = model(x)  # 使用PyTorch标准的模型调用方式获取输出
        # preds = preds.numpy()  # 如果后续还需要使用numpy相关的函数处理结果，可将torch张量转换回numpy数组
        # print(type(preds))
        if isinstance(preds, torch.Tensor):
            preds = preds.detach().cpu().numpy()
        if preds.ndim == 1:  # 如果preds原本是一维的，进行形状调整使其能适配赋值操作
            preds = np.resize(preds, (1, 10))  # 将preds调整为形状(1, 10)，这里可能需要根据实际情况合理填充数据，比如重复或截断等方式，需保证数据逻辑合理
            preds = torch.from_numpy(preds)
        elif preds.shape[1] != 10:  # 如果preds是多维但第二维度大小不是10，进行适当调整
            preds = preds[:, :10]  # 取前10列，使其第二维度大小变为10，同样需考虑数据截断是否符合逻辑
            preds = torch.from_numpy(preds)

        preds = preds.numpy()

        if preds.shape[1] == 10:  # 判断preds形状第二维度是否为10，若是则进行扩充
            new_preds = np.zeros((1, 1000))  # 创建形状为(1, 1000)的全零数组
            new_preds[:, :10] = preds  # 将原始preds数据填充到前10列
            preds = new_preds  # 更新preds为扩充后的形状

        print('Predicted:', decode_predictions(preds,1))  # 只显示top1
