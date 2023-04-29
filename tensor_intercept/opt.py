'''

OPT model training code

'''


import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

import sys

class OPT(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()
        self.input_linear = nn.Linear(28*28, d_model)  # Add this layer to handle input transformation
        self.attention = nn.MultiheadAttention(d_model, n_heads)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input_linear(x)  # Transform input using the added linear layer
        x = x.unsqueeze(1)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(1)
        x = self.dropout(x)
        x = self.feedforward(x)
        x = self.dropout(x)
        return x

def train(model, train_loader, optimizer, criterion, epochs):
    for epoch in range(epochs):
        for batch in train_loader:
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} loss: {loss.item()}")

def evaluate(model, val_loader, criterion):
    total_loss = 0
    for batch in val_loader:
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        pred = model(x)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        loss = criterion(pred, y)
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
        total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vocab_size = 10000
    d_model = 512
    n_heads = 8
    dim_feedforward = 2048
    dropout = 0.1

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transforms.ToTensor())
    val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transforms.ToTensor())

    # Create the data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=32, shuffle=False
    )

    # Create the model
    model = OPT(d_model, n_heads, dim_feedforward, dropout).to(device)
    print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)

    # Train the model
    for epoch in range(10):
        train(model, train_loader, optimizer, criterion, epochs=1)
        loss = evaluate(model, val_loader, criterion)
        print(f"Epoch {epoch} val loss: {loss}")
        print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)

    # Save the model
    torch.save(model.state_dict(), "model.pt")
    print(f"memory_allocated  = {torch.cuda.memory_allocated()}", file=sys.stderr)
