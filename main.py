import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=1, padding=(1, 1))

        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x) # [32, 8, 32, 32]
        x = F.relu(x)
        x = self.pool(x) # [32, 8, 16, 16]
        x = self.conv2(x) # [32, 16, 16, 16]
        x = F.relu(x)
        x = self.pool(x) # [32, 16, 8, 8]
        # x = self.conv3(x) # [32, 32, 8, 8]
        # x = F.relu(x)
        
        x = x.view(-1, 16 * 8 * 8)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x

BATCH_SIZE = 32
EPOCHS = 20
LR = 0.001

train_dataset = datasets.CIFAR10(root='./data/', train=True, download=True, transform=transforms.ToTensor())
test_dataset = datasets.CIFAR10(root='./data/', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# print("Using {}".format(device))
print(f"Using device : {device}")

model = CNN().to(device)
optim = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, optim, log_interval):
    model.train()
    for batch_idx, (image, label) in enumerate(train_loader):
        image = image.to(device)
        label = label.to(device)
        optim.zero_grad()
        output = model(image)
        loss = criterion(output, label)
        loss.backward()
        optim.step()

        if batch_idx % log_interval == 0:
            print(f"train Epoch: {Epoch} [{batch_idx * len(image)}/{len(train_loader.dataset)}({100. * batch_idx / len(train_loader):.0f}%)]\tTrain Loss: {loss.item()}")

def eval(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, test_accuracy

if __name__ == '__main__':
    for Epoch in range(1, EPOCHS + 1):
        train(model, train_loader, optim, log_interval=200)
        test_loss, test_accuracy = eval(model, test_loader)
        print(f"\n[EPOCH: {Epoch}]\tTest Loss: {test_loss:.4f}\tTest Accuracy: {test_accuracy} % \n")
