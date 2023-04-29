'''

Small model training code

'''


import torch
import torchvision
import torchvision.transforms as transforms
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

# Define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)  # Add a convolutional layer
        self.fc1 = torch.nn.Linear(28*28*6, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 28*28*6)
        x = self.fc1(x)
        return x

# Instantiate the network and move it to the GPU
net = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device, file=sys.stderr)
net.to(device)
print("net.to(device)", file=sys.stderr)
print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
print("", file=sys.stderr)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the network for one epoch with one step
for epoch in range(5):
    print(f"epoch = {epoch}", file=sys.stderr)
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # Get the inputs and move them to the GPU
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        print("inputs, labels = inputs.to(device), labels.to(device)", file=sys.stderr)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        print("", file=sys.stderr)

        # Zero the parameter gradients
        optimizer.zero_grad()
        print("optimizer.zero_grad()", file=sys.stderr)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        print("", file=sys.stderr)

        # Forward + backward + optimize
        outputs = net(inputs)
        print("outputs = net(inputs)", file=sys.stderr)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        print("", file=sys.stderr)

        loss = criterion(outputs, labels)
        print("loss = criterion(outputs, labels)", file=sys.stderr)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        print("", file=sys.stderr)

        loss.backward()
        print("loss.backward()", file=sys.stderr)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        print("", file=sys.stderr)

        optimizer.step()
        print("optimizer.step()", file=sys.stderr)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        print("", file=sys.stderr)

        # Print statistics
        running_loss += loss.item()
        print('loss: %.3f' % (running_loss), file=sys.stderr)
        running_loss = 0.0

print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
print('Finished Training', file=sys.stderr)

