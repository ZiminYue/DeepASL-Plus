#Source: https://datadrivenscience.com/hand-gesture-recognition-with-pytorch-a-comprehensive-tutorial/
# Import necessary packages
import os
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load the data from my dataset
# class LeapGestRecog(Dataset):
#     def __init__(self, root, transform=None):
#         self.root = root
#         self.transform = transform
#         self.x = []
#         self.y = []

#         folders = os.listdir(root)
#         for folder in folders:
#             for dirpath, dirnames, filenames in os.walk(os.path.join(root, folder)):
#                 for filename in filenames:
#                     self.x.append(os.path.join(dirpath, filename))
#                     self.y.append(folder)
#         self.len = len(self.x)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, index):
#         img = Image.open(self.x[index]).convert('L')
#         y = self.y[index]
#         if self.transform:
#             img = self.transform(img)
#         return img, y

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # 调整图片大小为128x128
    transforms.Grayscale(num_output_channels=1),  # 转为灰度图像
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化
])

# 使用 ImageFolder 加载数据集
root = './My Model/aslDataSet4Training'  # 修改为你的数据集路径
root_train = './My Model/aslDataSet4Training/train'  # 训练集路径
root_test = './My Model/aslDataSet4Training/val'  # 验证集路径

# 加载训练集
dataset = datasets.ImageFolder(root=root, transform=transform)

# 使用 ImageFolder 加载训练集和验证集
train_dataset = datasets.ImageFolder(root=root_train, transform=transform)
test_dataset = datasets.ImageFolder(root=root_test, transform=transform)


# 创建 DataLoader
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# 获取类别标签
print(dataset.classes)  # 输出类别名称（即文件夹名称）


# Building the Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.fc1 = nn.Linear(128 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, 36)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)  # 打印第一个卷积层输出的形状
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)  # 打印第二个卷积层输出的形状

        x = F.relu(self.conv3(x))
        # print(x.shape)  # 打印第三个卷积层输出的形状

        x = x.view(x.size(0), -1)  # 展平操作
        # print(x.shape)  # 打印展平后的形状
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()

# # Training the Model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model.to(device)

# optimizer = optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.CrossEntropyLoss()

# best_loss = 100000
# num_epochs = 10

# # Training loop
# for epoch in range(num_epochs): 
#     train_loss = 0.0
#     model.train()  # 设置模型为训练模式

#     # 训练阶段
#     for i, (inputs, labels) in enumerate(train_loader):
#         inputs, labels = inputs.to(device), labels.to(device)
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()  # 累积每个batch的损失

#     # 计算训练损失的平均值
#     train_loss /= len(train_loader)

#     test_loss = 0.0
#     model.eval()  # 设置模型为评估模式
#     with torch.no_grad():  # 在测试时不计算梯度
#         for i, (inputs, labels) in enumerate(test_loader):
#             inputs, labels = inputs.to(device), labels.to(device)
#             outputs = model(inputs)
#             loss = loss_fn(outputs, labels)
#             test_loss += loss.item()  # 累积每个batch的损失

#     # 计算测试损失的平均值
#     test_loss /= len(test_loader)

#     # 每10个epoch输出一次损失
#     if (epoch + 1) % 10 == 0:
#         print(f'Epoch {epoch + 1}, train loss: {train_loss:.3f}, test loss: {test_loss:.3f}')

#     # 保存最优模型
#     if test_loss < best_loss:
#         best_loss = test_loss
#         torch.save(model.state_dict(), 'aslgettyHands.pth')  # 保存最优模型

# Training the Model

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item()))

for epoch in range(1, 7):
    train(epoch)

# Testing the Model
def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

test()

# 保存训练好的模型
torch.save(model.state_dict(), 'gettyHands.pth')
