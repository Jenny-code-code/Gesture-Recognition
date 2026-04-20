import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_ds = datasets.ImageFolder("./train", transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)

test_ds = datasets.ImageFolder("./test", transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 16, 3)
        self.fc1 = nn.Linear(61504, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

def imshow(img):
    npimg = img.numpy()
    plt.imshow(npimg[0].transpose(1, 2, 0))
    plt.show()    


image, label = next(iter(train_loader))
imshow(image)
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
LOSS = 0
for epoch in range(20):
    LOSS = 0
    for i, data in enumerate(train_loader, 0):
        image, label = data
        image = image.to(device)
        label = label.to(device)
        out = model(image)

        optimizer.zero_grad()
        loss = criterion(out, label)
        LOSS += loss.item()
        loss.backward()
        optimizer.step()
        if (i + 1) % 200 == 0:
            print(f"epoch: {i+1} loss: {LOSS / 200:.4f}")
            LOSS = 0
    total = len(test_ds)
    correct = 0
    for i, data in enumerate(test_loader):
        image, label = data
        image = image.to(device)
        label = label.to(device)
        out = model(image)
        if torch.argmax(out) == label:
            correct += 1
    print(f"Accuracy: {correct / total:.4f}")

for i in range(10):
    image, label = next(iter(test_loader))
    npimg = image
    image = image.to(device)
    label = label.to(device)
    out = model(image)
    print(torch.argmax(out))
    imshow(npimg)
        