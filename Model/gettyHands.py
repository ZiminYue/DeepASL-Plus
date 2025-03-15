#Reference: https://datadrivenscience.com/hand-gesture-recognition-with-pytorch-a-comprehensive-tutorial/
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Data preprocess
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # resize image to 128x128
    transforms.Grayscale(num_output_channels=1),  # convert to grayscale
    transforms.ToTensor(),  # convert to Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

# Load dataset with ImageFolder 

root_train = './My Model/aslDataSet4Training/train'  
root_test = './My Model/aslDataSet4Training/val'  


train_dataset = datasets.ImageFolder(root=root_train, transform=transform)
test_dataset = datasets.ImageFolder(root=root_test, transform=transform)


# Create DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



# Build the Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 29 * 29, 128)
        self.fc2 = nn.Linear(128, 36)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)  
        
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        # print(x.shape)  

        x = x.view(x.size(0), -1) 
        # print(x.shape)  
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()

# Train the Model

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

for epoch in range(1, 10):
    train(epoch)

# Test the Model
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
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))

test()

# Save the model
torch.save(model.state_dict(), 'aslgettyHands.pth')
