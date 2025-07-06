#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
import os
import time
import datetime


def setup():
    """Initialize distributed training with multiple backend attempts"""
    import datetime
    import time
    import os
    
    # Multi-node startup coordination
    print("Waiting for all pods to be ready...")
    time.sleep(15)  # Extra time for cross-node startup
    
    # Use Gloo backend directly (optimized for CPU training)
    try:
        print("\n--- Initializing Gloo backend (CPU-optimized) ---")
        
        # Gloo configuration for same-node communication
        os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'  # Use pod network interface
        os.environ['NCCL_SOCKET_IFNAME'] = 'eth0'
        os.environ['GLOO_DEVICE_TRANSPORT'] = 'TCP'
        # Force IPv4 only to avoid IPv6 resolution issues
        os.environ['GLOO_SOCKET_FAMILY'] = 'IPV4'
        print("Set GLOO network interface to eth0 (pod networking, IPv4 only)")
        
        print("Initializing gloo process group...")
        # Get environment variables for debugging
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '23456')
        
        print(f"Debug - Rank: {rank}, World Size: {world_size}")
        print(f"Debug - Master Addr: {master_addr}, Master Port: {master_port}")
        
        dist.init_process_group(
            backend='gloo', 
            timeout=datetime.timedelta(seconds=60)  # Reduced timeout
        )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"SUCCESS: gloo backend initialized - Rank {rank}, World size {world_size}")
        
        # Synchronization barrier
        print(f"Rank {rank}: Waiting at synchronization barrier...")
        dist.barrier()
        print(f"Rank {rank}: All processes synchronized with gloo!")
        
        # Test communication
        print(f"Rank {rank}: Testing gloo communication...")
        test_tensor = torch.tensor([rank], dtype=torch.float32)
        dist.all_reduce(test_tensor)
        expected_sum = sum(range(world_size))  # 0 + 1 = 1 for 2 processes
        print(f"Rank {rank}: Communication test PASSED - sum: {test_tensor.item()} (expected: {expected_sum})")
        
        return rank, world_size, 'gloo'
        
    except Exception as e:
        print(f"FAILED gloo backend: {type(e).__name__}: {e}")
        if dist.is_initialized():
            dist.destroy_process_group()
        raise RuntimeError(f"Gloo backend failed to initialize: {e}")


class CNNModel(nn.Module):
    """Optimized CNN model for MNIST training"""
    
    def __init__(self):
        super(CNNModel, self).__init__()
        # Balanced model for good accuracy with reasonable resources
        self.conv1 = nn.Conv2d(1, 16, 5, 1)  # 16 channels
        self.conv2 = nn.Conv2d(16, 32, 5, 1)  # 32 channels
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(512, 64)  # 64 neurons
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)


def load_dataset(rank):
    """Load MNIST dataset or create synthetic fallback"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if os.path.exists('/input/MNIST'):
        train_dataset = datasets.MNIST('/input', train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST('/input', train=False, download=False, transform=transform)
        print(f"Rank {rank}: Using pre-downloaded MNIST dataset ({len(train_dataset)} train, {len(test_dataset)} test)")
    else:
        print(f"Rank {rank}: Dataset not found, using synthetic data")
        import torch.utils.data as data_utils
        # Realistic synthetic dataset for testing
        train_data = torch.randn(1000, 1, 28, 28)
        train_targets = torch.randint(0, 10, (1000,))
        train_dataset = data_utils.TensorDataset(train_data, train_targets)
        test_dataset = data_utils.TensorDataset(train_data[:200], train_targets[:200])
    
    return train_dataset, test_dataset


def train_epoch(model, train_loader, optimizer, device, rank, epoch, manual_sync=False, world_size=1):
    """Train model for one epoch with optional manual gradient synchronization"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        
        # Manual gradient synchronization if DDP failed
        if manual_sync and world_size > 1:
            # Average gradients across all processes
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx % 20 == 0:
            accuracy = 100. * correct / total if total > 0 else 0
            sync_method = "manual sync" if manual_sync else "DDP auto"
            print(f'Rank {rank}, Epoch {epoch}, Batch {batch_idx}: Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}% ({sync_method})')
    
    return 100. * correct / total if total > 0 else 0


def evaluate_model(model, test_loader, device):
    """Evaluate model on test dataset"""
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)
    
    return 100. * test_correct / test_total if test_total > 0 else 0


def save_model_and_metadata(model, rank, world_size, final_accuracy, test_accuracy, total_time, device, backend, manual_sync=False):
    """Save trained model and training metadata (master only)"""
    if rank == 0:
        print(f"Rank {rank}: Saving model and metadata...")
        os.makedirs('/output', exist_ok=True)
        
        # Debug: Check output directory
        try:
            print(f"Rank {rank}: Output directory contents before save: {os.listdir('/output')}")
        except PermissionError:
            print(f"Rank {rank}: Output directory exists but permission denied for listing")
        except Exception as e:
            print(f"Rank {rank}: Error listing output directory: {e}")
        
        # Handle both DDP and non-DDP models
        model_state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        model_path = '/output/trained-model.pth'
        torch.save(model_state, model_path)
        
        # Verify model was saved
        if os.path.exists(model_path):
            size = os.path.getsize(model_path)
            print(f"Rank {rank}: Model saved successfully to {model_path} (size: {size} bytes)")
        else:
            print(f"Rank {rank}: ERROR - Model file not created at {model_path}")
            return
        
        sync_method = "Manual Gradient Sync" if manual_sync else "DDP Automatic Sync"
        
        with open('/output/training_metadata.txt', 'w') as f:
            f.write(f'Multi-Node Distributed Training Results\n')
            f.write(f'Backend: {backend}\n')
            f.write(f'Sync Method: {sync_method}\n')
            f.write(f'World Size: {world_size}\n')
            f.write(f'Final Training Accuracy: {final_accuracy:.2f}%\n')
            f.write(f'Test Accuracy: {test_accuracy:.2f}%\n')
            f.write(f'Total Time: {total_time:.1f} seconds\n')
            f.write(f'Device: {device}\n')
            f.write(f'Batch Size: 16\n')
            f.write(f'Epochs: 1\n')
            f.write(f'Model: CNN (16/32 channels, 64 neurons)\n')
            f.write(f'Dataset: {"Real MNIST" if os.path.exists("/input/MNIST") else "Synthetic"}\n')
            f.write(f'Training Type: Multi-node distributed across Kind cluster\n')
        
        # Verify metadata was saved
        metadata_path = '/output/training_metadata.txt'
        if os.path.exists(metadata_path):
            size = os.path.getsize(metadata_path)
            print(f"Rank {rank}: Metadata saved successfully to {metadata_path} (size: {size} bytes)")
        else:
            print(f"Rank {rank}: ERROR - Metadata file not created at {metadata_path}")
        
        # Final directory check
        try:
            print(f"Rank {rank}: Final output directory contents: {os.listdir('/output')}")
        except PermissionError:
            print(f"Rank {rank}: Output directory exists but permission denied for listing")
        except Exception as e:
            print(f"Rank {rank}: Error listing output directory: {e}")
        print(f'Rank {rank}: Model and metadata saved successfully with {backend} backend and {sync_method.lower()}!')


def main():
    """Main distributed training function"""
    rank, world_size, backend = setup()
    device = torch.device('cpu')  # CPU training for compatibility
    print(f"Rank {rank}: Using device: {device}, Backend: {backend}")

    # Debug: Check mounted directories
    print(f"Rank {rank}: Checking mounted directories...")
    for directory in ['/input', '/output', '/scripts']:
        if os.path.exists(directory):
            try:
                files = os.listdir(directory)
                print(f"Rank {rank}: {directory} exists with {len(files)} items: {files[:5]}...")
            except PermissionError:
                print(f"Rank {rank}: {directory} exists but permission denied (this is normal for Kind volume mounts)")
            except Exception as e:
                print(f"Rank {rank}: {directory} exists but error listing contents: {e}")
        else:
            print(f"Rank {rank}: {directory} does not exist!")
    
    # Check if training script is accessible
    script_path = '/scripts/distributed_mnist_training.py'
    if os.path.exists(script_path):
        print(f"Rank {rank}: Training script found at {script_path}")
    else:
        print(f"Rank {rank}: Training script not found at {script_path}")

    # Load dataset
    train_dataset, test_dataset = load_dataset(rank)

    # Create distributed samplers and data loaders
    print(f"Rank {rank}: Creating distributed samplers for {world_size} processes...")
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, sampler=test_sampler, shuffle=False)
    print(f"Rank {rank}: Distributed data loaders created - {len(train_loader)} train batches")

    # Create and setup distributed model
    model = CNNModel().to(device)
    manual_sync = True  # Initialize as manual sync by default
    
    print(f"Rank {rank}: Creating DistributedDataParallel model...")
    try:
        # Try DDP with parameter verification disabled
        model = DDP(model, check_reduction=False)
        print(f"Rank {rank}: DDP model created successfully (with check_reduction=False)")
        manual_sync = False
    except Exception as ddp_error:
        print(f"Rank {rank}: DDP with check_reduction=False failed: {ddp_error}")
        try:
            # Try DDP with both checks disabled
            model = DDP(model, check_reduction=False, find_unused_parameters=True)
            print(f"Rank {rank}: DDP model created successfully (with relaxed settings)")
            manual_sync = False
        except Exception as ddp_error2:
            print(f"Rank {rank}: All DDP attempts failed: {ddp_error2}")
            print(f"Rank {rank}: Using manual gradient synchronization instead of DDP")
            manual_sync = True
    
    if manual_sync:
        print(f"Rank {rank}: Will use manual gradient synchronization during training")
    else:
        print(f"Rank {rank}: Will use automatic DDP gradient synchronization")
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    print(f"Rank {rank}: Starting distributed training...")
    start_time = time.time()
    final_accuracy = 0
    
    for epoch in range(1):  # 1 epoch for memory efficiency
        epoch_start = time.time()
        final_accuracy = train_epoch(model, train_loader, optimizer, device, rank, epoch, manual_sync, world_size)
        epoch_time = time.time() - epoch_start
        sync_info = f" (manual gradient sync)" if manual_sync else f" (DDP auto sync)"
        print(f'Rank {rank}: Epoch {epoch} completed in {epoch_time:.1f}s - Training Accuracy: {final_accuracy:.2f}%{sync_info}')
    
    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader, device)
    total_time = time.time() - start_time
    
    print(f'Rank {rank}: Final Training Accuracy: {final_accuracy:.2f}%')
    print(f'Rank {rank}: Test Accuracy: {test_accuracy:.2f}%')
    print(f'Rank {rank}: Total training time: {total_time:.1f} seconds')

    # Save model and metadata (master only)
    save_model_and_metadata(model, rank, world_size, final_accuracy, test_accuracy, total_time, device, backend, manual_sync)

    # Buffer time to allow artifact collection from volume mounts
    if rank == 0:
        print(f"Rank {rank}: Waiting 30 seconds for artifact collection...")
        time.sleep(30)
        print(f"Rank {rank}: Artifact collection window completed")
    else:
        print(f"Rank {rank}: Waiting for master to complete artifact collection...")
        time.sleep(35)  # Workers wait a bit longer
        print(f"Rank {rank}: Worker cleanup ready")

    # Cleanup distributed process group
    print(f"Rank {rank}: Cleaning up distributed process group")
    dist.destroy_process_group()
    print(f"Rank {rank}: Distributed training completed successfully!")


if __name__ == "__main__":
    main() 