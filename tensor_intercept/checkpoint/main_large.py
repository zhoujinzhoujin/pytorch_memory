'''

Large model training code with checkpoints

'''


import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import sys

# Define the transforms to apply to the data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Download and load the CIFAR-10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                          shuffle=True, num_workers=2)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(32*32*3, 1024)
        self.fc2 = torch.nn.Linear(1024, 4096)
        self.fc3 = torch.nn.Linear(4096, 4096)
        self.fc4 = torch.nn.Linear(4096, 4096)
        self.fc5 = torch.nn.Linear(4096, 4096)
        self.fc6 = torch.nn.Linear(4096, 4096)
        self.fc7 = torch.nn.Linear(4096, 1024)
        self.fc8 = torch.nn.Linear(1024, 512)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = checkpoint(self.fc1, x)
        x = checkpoint(self.fc2, x)
        x = checkpoint(self.fc3, x)
        x = checkpoint(self.fc4, x)
        x = checkpoint(self.fc5, x)
        x = checkpoint(self.fc6, x)
        x = checkpoint(self.fc7, x)
        x = self.fc8(x)
        return x


    
# Instantiate the network and move it to the GPU
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network for one epoch with one step
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and move them to the GPU
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        inputs.requires_grad = True
        optimizer.zero_grad()

        outputs = net(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        # Print statistics
        running_loss += loss.item()
        # print(running_loss)
        running_loss = 0.0




