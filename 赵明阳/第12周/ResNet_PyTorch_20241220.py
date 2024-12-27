import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions

class identity_block(nn.Module):
    def __init__(self, in_channel, outs, kernerl_size, stride, padding):
        super(identity_block, self).__init__()
        assert len(outs) == 3, "outs参数长度应为3"
        assert len(kernerl_size) == 3, "kernel_size参数长度应为3"
        assert len(stride) == 4, "stride参数长度应为4"
        assert len(padding) == 3, "padding参数长度应为3"
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernerl_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernerl_size[1], stride=stride[0], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernerl_size[2], stride=stride[0], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(self.bn1(out))

        out = self.conv2(out)
        out = F.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        return F.relu(out + x)


class conv_block(nn.Module):
    def __init__(self, in_channel, outs, kernel_size, stride, padding):
        super(conv_block, self).__init__()
        # out1, out2, out3 = outs
        # print(outs)
        self.conv1 = nn.Conv2d(in_channel, outs[0], kernel_size=kernel_size[0], stride=stride[0], padding=padding[0])
        self.bn1 = nn.BatchNorm2d(outs[0])
        self.conv2 = nn.Conv2d(outs[0], outs[1], kernel_size=kernel_size[1], stride=stride[1], padding=padding[1])
        self.bn2 = nn.BatchNorm2d(outs[1])
        self.conv3 = nn.Conv2d(outs[1], outs[2], kernel_size=kernel_size[2], stride=stride[2], padding=padding[2])
        self.bn3 = nn.BatchNorm2d(outs[2])

        self.extra = nn.Sequential(
            nn.Conv2d(in_channel, outs[2], kernel_size=1, stride=stride[3], padding=0),
            nn.BatchNorm2d(outs[2])
        )

    def forward(self, x):
        x_shortcut = self.extra(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        return F.relu(x_shortcut + out)

#定义网络结构：使用 PyTorch 框架定义了 ResNet50 网络结构，包含了基础的卷积层、最大池化层以及由 conv_block 和 identity_block
# 组成的多个层（layer1 - layer4），最后还有平均池化层以及用卷积替代全连接的层，整体结构遵循经典的 ResNet50 架构，用于图像特征
# 提取和分类任务。
class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            conv_block(64, outs=[64, 64, 256], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            identity_block(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
            identity_block(256, outs=[64, 64, 256], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0]),
        )

        self.layer2 = nn.Sequential(
            conv_block(256, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2], padding=[0, 1, 0]),
            identity_block(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            identity_block(512, outs=[128, 128, 512], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            conv_block(512, outs=[128, 128, 512], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1], padding=[0, 1, 0])
        )

        self.layer3 = nn.Sequential(
            conv_block(512, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
                              padding=[0, 1, 0]),
            identity_block(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            identity_block(1024, outs=[256, 256, 1024], kernerl_size=[1, 3, 1], stride=[1, 1, 1, 1],
                               padding=[0, 1, 0]),
            conv_block(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            conv_block(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            conv_block(1024, outs=[256, 256, 1024], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0])
        )

        self.layer4 = nn.Sequential(
            conv_block(1024, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 2, 1, 2],
                              padding=[0, 1, 0]),
            conv_block(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0]),
            conv_block(2048, outs=[512, 512, 2048], kernel_size=[1, 3, 1], stride=[1, 1, 1, 1],
                              padding=[0, 1, 0])
        )

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, ceil_mode=False)
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc = nn.Linear(2048, 10)
        # 使用卷积代替全连接
        self.conv11 = nn.Conv2d(2048, 10, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = self.conv11(out)
        out = out.reshape(x.shape[0], -1)
        # out = self.fc(out)
        return out

    def predict(self, x):
        """
        自定义的预测方法，用于对输入数据进行预测
        :param x: 输入数据，通常是张量形式，形状需符合模型输入要求
        :return: 模型的预测结果
        """
        self.eval()
        with torch.no_grad():
            x = x.copy()
            x = torch.from_numpy(x).float()
            output = self(x)
            return output
#定义数据集类：创建了 ImageDataset 类，继承自 torch.utils.data.Dataset，用于加载图像数据，在 __getitem__ 方法中实现了读取
# 图像、应用图像变换（如调整大小、转换为张量、归一化等）的功能，通过 __len__ 方法返回数据集的大小（即图像数量）。
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

#主程序逻辑：在 if __name__ == '__main__': 部分，实例化了 ResNet50 模型、定义了图像预处理操作、创建了数据集和数据加载器
# （DataLoader），然后将模型设置为评估模式，通过循环数据加载器中的批次数据，使用模型进行预测，尝试获取并处理预测类别索引，
# 但部分相关代码被注释掉了，整体是一个简单的图像分类推理流程的框架。
if __name__ == '__main__':
    # 创建ResNet50模型实例
    model = ResNet50()

    # 图像预处理操作，与Keras中类似的功能，将图像调整大小、转换为张量并进行归一化等
    # 假设多张图片路径列表
    img_paths = ['elephant.jpg', 'bike.jpg']
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    batch_size = 32
    # 创建数据集实例
    dataset = ImageDataset(img_paths, transform)
    # 创建数据集实例
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)
    #
    # 将模型设置为评估模式
    model.eval()
    with torch.no_grad():
        # 使用模型进行预测
        for imgs in data_loader:
            output = model(imgs)
            # 模拟获取预测类别（实际应用中可根据具体任务进行后处理，如取概率最大的类别等）
            _, predicted = torch.max(output, 1)
            # predicted = predicted.squeeze(1)  # 通过squeeze操作去掉维度为1的维度，使其变为一维张量
            # print(f'预测类别索引: {predicted.tolist()}')  # 使用tolist()方法将一维张量转换为Python列表，方便打印展示等操作
            if predicted.dim() == 1:
                print(f'预测类别索引: {predicted.tolist()}')
            else:
                # 假设是二维且第二维大小为1的情况，进行squeeze操作并处理
                predicted = predicted.squeeze(1)
                print(f'预测类别索引: {predicted.tolist()}')
    summary(model, input_size=(3, 224, 224))  # 这里假设输入图像的通道数是3，尺寸是224x224，根据实际情况调整输入尺寸参数
    # 接着指定了一个图像文件路径（示例中先是 'elephant.jpg'，也可以替换为其他图像文件路径），通过 image.load_img 函数读取图像并
    # 调整其大小为 (224, 224)，符合模型输入要求，然后使用 image.img_to_array 将图像转换为数组形式，再通过 np.expand_dims 增加
    # 一个维度（模拟一个批次的数据，因为模型输入通常期望是包含批次维度的张量形式，即 (batch_size, height, width, channels)），
    img_path = 'elephant.jpg'
    # img_path = 'bike.jpg'
    #img = load_img(img_path, target_size=(224, 224))
    #img = Image.load_img(img_path, target_size=(224, 224))
    img = Image.open(img_path)
    img = img.resize((224, 224))  # 使用resize方法调整图像大小
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # 添加这行代码，将numpy.ndarray类型的x转换为torch.Tensor类型
    x = torch.from_numpy(x).float()  # 注意这里最好转换为float类型，符合模型中数据类型的常见要求，可根据实际情况调整类型
    # 最后使用 preprocess_input 对图像数据进行预处理（例如归一化等操作，使其符合模型训练时对输入数据的要求）。
    print(f"输入preprocess_input函数之前x的维度信息: {x.ndim}，形状: {x.shape}，数据类型: {x.dtype}")
    x = np.transpose(x, (0,3,1,2))  # 将维度顺序调整为(batch_size, height, width, channels)
    x = x.detach().numpy()
    print(f"输入detach函数之后x的维度信息: {x.ndim}，形状: {x.shape}，数据类型: {x.dtype}")
    x = preprocess_input(x)

    # 之后将处理好的图像数据输入到模型中，通过 model.predict(x) 进行预测，得到预测结果 preds，再使用 decode_predictions 函数对
    # 预测结果进行解码，将其转换为更易读的形式（比如以包含类别名称、类别编号和对应概率的列表形式呈现），最后打印出预测结果，展示模型
    # 对输入图像的分类预测情况。
    # 创建一个全零张量，形状为(1, 1000)，用于填充扩展

    print('Input image shape:', x.shape)
    preds = model.predict(x)
    new_preds = torch.zeros((1, 1000))
    new_preds[:, :10] = preds  # 将原来的preds数据填充到新张量的前10列，对应实际的10个分类
    new_preds = new_preds.detach().numpy()  # 将torch.Tensor类型转换为numpy.ndarray类型
    print(f"new_preds的数据类型: {type(new_preds)}")
    # print(f"new_preds的维度信息: {new_preds.ndim}，形状: {new_preds.shape}")
    # print(f"new_preds的部分数据示例: {new_preds[:5]}")  # 打印部分数据示例，查看数值情况
    print('Predicted:', decode_predictions(new_preds))