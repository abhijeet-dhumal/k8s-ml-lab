#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

class SimpleMNIST(nn.Module):
    def __init__(self):
        super(SimpleMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 100 == 0:
            accuracy = 100. * correct / total
            print(f'Epoch {epoch}, Batch {batch_idx}: Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%')
    
    epoch_accuracy = 100. * correct / total
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch} completed: Average Loss: {avg_loss:.6f}, Accuracy: {epoch_accuracy:.2f}%')
    return epoch_accuracy

def test_model(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    print(f'Test Results: Average Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return accuracy

def main():
    print("🚀 Starting Simple Single-Pod MNIST Training")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data loading
    print("Loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Check if data exists
    data_path = '/input'
    if not os.path.exists(data_path):
        print(f"Data directory {data_path} not found, using /tmp/input")
        data_path = '/tmp/input'
    
    train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    # Model, optimizer
    model = SimpleMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training
    print("Starting training...")
    start_time = time.time()
    
    for epoch in range(1, 3):  # Train for 2 epochs
        train_accuracy = train_epoch(model, device, train_loader, optimizer, epoch)
        test_accuracy = test_model(model, device, test_loader)
        
        if train_accuracy > 95:
            print(f"Early stopping: achieved {train_accuracy:.2f}% accuracy")
            break
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    model_path = '/app/output/simple_mnist_model.pth'
    os.makedirs('/app/output', exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save metadata
    metadata_path = '/app/output/training_metadata.txt'
    with open(metadata_path, 'w') as f:
        f.write(f"Training completed successfully\n")
        f.write(f"Final test accuracy: {test_accuracy:.2f}%\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Device used: {device}\n")
        f.write(f"Model parameters: {sum(p.numel() for p in model.parameters())}\n")
    
    print(f"Training metadata saved to {metadata_path}")
    print("🎉 Training completed successfully!")

if __name__ == '__main__':
    main() 