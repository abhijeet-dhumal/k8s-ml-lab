# 🚀 Example 08: GPU Training

**Goal:** Enable GPU acceleration for distributed PyTorch training.

**Prerequisites:**
- Working cluster with GPU support
- NVIDIA device plugin installed
- Custom dataset example (07) completed

**What you'll learn:**
- GPU resource allocation
- CUDA optimization
- Multi-GPU distributed training
- GPU monitoring and debugging

**Estimated time:** 30 minutes

---

## Overview

This example demonstrates how to leverage GPUs for faster distributed training. We'll cover both single-GPU and multi-GPU scenarios.

## GPU Cluster Types

| Cluster Type | GPU Support | Setup Method |
|-------------|-------------|--------------|
| **Kind** | Limited | NVIDIA Container Toolkit |
| **EKS** | Native | GPU AMI + Node Groups |
| **GKE** | Native | GPU Node Pools |
| **AKS** | Native | GPU VM SKUs |
| **On-premises** | Full | NVIDIA Device Plugin |

## Step 1: Verify GPU Availability

```bash
# Check cluster GPU capacity
kubectl describe nodes | grep -A 5 "Capacity:" | grep nvidia

# Check available GPU resources
kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu"

# List GPU device plugin pods
kubectl get pods -n kube-system | grep nvidia
```

## Step 2: Create GPU-Optimized Training Script

```bash
# Create GPU training script
cat > scripts/distributed_gpu_training.py << 'EOF'
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import time

class GPUOptimizedNet(nn.Module):
    """GPU-optimized CNN with mixed precision support"""
    def __init__(self, num_classes=10, channels=1):
        super(GPUOptimizedNet, self).__init__()
        # Larger model for GPU efficiency
        self.conv1 = nn.Conv2d(channels, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
    def forward(self, x):
        x = self.pool(self.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(self.relu(self.batch_norm2(self.conv2(x))))
        x = self.pool(self.relu(self.batch_norm3(self.conv3(x))))
        x = self.pool(self.relu(self.batch_norm4(self.conv4(x))))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def setup_distributed():
    """Initialize distributed training"""
    # Set CUDA device before initializing process group
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        print(f"Using GPU {local_rank}: {torch.cuda.get_device_name(local_rank)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
    else:
        print("WARNING: CUDA not available, falling back to CPU")
    
    # Initialize process group
    if torch.cuda.is_available():
        dist.init_process_group(backend='nccl')  # NCCL for GPU
    else:
        dist.init_process_group(backend='gloo')  # Gloo fallback
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    print(f"SUCCESS: {backend} backend initialized - Rank {rank}, World size {world_size}")
    
    return rank, world_size, local_rank

def load_dataset(rank, dataset_type='mnist'):
    """Load dataset with GPU-optimized transforms"""
    # Enhanced transforms for GPU training
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST stats
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if dataset_type == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root='/input/MNIST', 
            train=True, 
            transform=train_transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='/input/MNIST', 
            train=False, 
            transform=test_transform
        )
        return train_dataset, test_dataset, 10, 1
    
    else:
        raise ValueError(f"Dataset type {dataset_type} not implemented for GPU training")

def main():
    # Training parameters
    dataset_type = os.environ.get('DATASET_TYPE', 'mnist')
    epochs = int(os.environ.get('EPOCHS', '20'))
    batch_size = int(os.environ.get('BATCH_SIZE', '128'))  # Larger batch for GPU
    learning_rate = float(os.environ.get('LEARNING_RATE', '0.001'))
    use_mixed_precision = os.environ.get('MIXED_PRECISION', 'true').lower() == 'true'
    
    # Setup distributed training
    rank, world_size, local_rank = setup_distributed()
    
    # Set device
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"Training configuration:")
        print(f"  Device: {device}")
        print(f"  Dataset: {dataset_type}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Mixed precision: {use_mixed_precision}")
        print(f"  World size: {world_size}")
    
    # Load dataset
    train_dataset, test_dataset, num_classes, channels = load_dataset(rank, dataset_type)
    
    # Create data loaders with pin_memory for GPU efficiency
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,  # More workers for GPU
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True
    )
    
    # Create model and move to device
    model = GPUOptimizedNet(num_classes=num_classes, channels=channels)
    model = model.to(device)
    
    # Setup DDP
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = DDP(model)
    
    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_mixed_precision and torch.cuda.is_available() else None
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if rank == 0 and batch_idx % 50 == 0:
                accuracy = 100. * correct / total
                lr = optimizer.param_groups[0]['lr']
                gpu_memory = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
                print(f"Epoch {epoch:2d}, Batch {batch_idx:3d}: "
                      f"Loss: {loss.item():.4f}, Acc: {accuracy:5.1f}%, "
                      f"LR: {lr:.6f}, GPU Mem: {gpu_memory:.1f}GB")
        
        scheduler.step()
        
        if rank == 0:
            epoch_time = time.time() - epoch_start
            train_accuracy = 100. * correct / total
            print(f"Epoch {epoch:2d} completed in {epoch_time:.1f}s, "
                  f"Training Accuracy: {train_accuracy:.2f}%")
    
    # Test evaluation
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    if rank == 0:
        test_accuracy = 100. * correct / total
        avg_test_loss = test_loss / len(test_loader)
        print(f"Test Results: Accuracy: {test_accuracy:.2f}%, Loss: {avg_test_loss:.4f}")
        
        # Save model
        os.makedirs('/app/models', exist_ok=True)
        model_path = f'/output/trained-model.pth'
        torch.save(model.module.state_dict(), model_path)
        
        # Save training metadata
        metadata = f"""Dataset: {dataset_type}
Model: GPUOptimizedNet
Classes: {num_classes}
Channels: {channels}
Training Accuracy: {train_accuracy:.2f}%
Test Accuracy: {test_accuracy:.2f}%
Test Loss: {avg_test_loss:.4f}
Epochs: {epochs}
Batch Size: {batch_size}
Learning Rate: {learning_rate}
Mixed Precision: {use_mixed_precision}
World Size: {world_size}
Device: {device}
Total Parameters: {total_params:,}
"""
        with open(f'/output/training_metadata.txt', 'w') as f:
            f.write(metadata)
        
        print("✅ GPU model saved successfully!")
        
        # Print GPU memory summary
        if torch.cuda.is_available():
            print(f"GPU Memory Summary:")
            print(f"  Allocated: {torch.cuda.memory_allocated(device) / 1e9:.2f} GB")
            print(f"  Cached: {torch.cuda.memory_reserved(device) / 1e9:.2f} GB")

if __name__ == "__main__":
    main()
EOF
```

## Step 3: Create GPU Job Configuration

```bash
# Create GPU job configuration
cat > configs/pytorch-gpu-job.yaml << 'EOF'
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-gpu-distributed
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
            command:
            - python3
            - /app/distributed_gpu_training.py
            env:
            - name: DATASET_TYPE
              value: "cifar10"
            - name: EPOCHS
              value: "20"
            - name: BATCH_SIZE
              value: "128"
            - name: LEARNING_RATE
              value: "0.001"
            - name: MIXED_PRECISION
              value: "true"
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            resources:
              requests:
                cpu: 4
                memory: 8Gi
                nvidia.com/gpu: 1
              limits:
                cpu: 8
                memory: 16Gi
                nvidia.com/gpu: 1
            volumeMounts:
            - name: training-script
              mountPath: /app
            - name: dataset-volume
              mountPath: /data
            - name: model-volume
              mountPath: /app/models
            - name: shm-volume
              mountPath: /dev/shm
          volumes:
          - name: training-script
            configMap:
              name: pytorch-training-script
          - name: dataset-volume
            hostPath:
              path: /tmp/data
              type: DirectoryOrCreate
          - name: model-volume
            hostPath:
              path: /tmp/models
              type: DirectoryOrCreate
          - name: shm-volume
            emptyDir:
              medium: Memory
              sizeLimit: 2Gi
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
            command:
            - python3
            - /app/distributed_gpu_training.py
            env:
            - name: DATASET_TYPE
              value: "cifar10"
            - name: EPOCHS
              value: "20"
            - name: BATCH_SIZE
              value: "128"
            - name: LEARNING_RATE
              value: "0.001"
            - name: MIXED_PRECISION
              value: "true"
            - name: CUDA_VISIBLE_DEVICES
              value: "0"
            resources:
              requests:
                cpu: 4
                memory: 8Gi
                nvidia.com/gpu: 1
              limits:
                cpu: 8
                memory: 16Gi
                nvidia.com/gpu: 1
            volumeMounts:
            - name: training-script
              mountPath: /app
            - name: dataset-volume
              mountPath: /data
            - name: model-volume
              mountPath: /app/models
            - name: shm-volume
              mountPath: /dev/shm
          volumes:
          - name: training-script
            configMap:
              name: pytorch-training-script
          - name: dataset-volume
            hostPath:
              path: /tmp/data
              type: DirectoryOrCreate
          - name: model-volume
            hostPath:
              path: /tmp/models
              type: DirectoryOrCreate
          - name: shm-volume
            emptyDir:
              medium: Memory
              sizeLimit: 2Gi
EOF
```

## Step 4: Setup GPU Cluster (if needed)

### For Cloud Clusters:

```bash
# EKS with GPU nodes
eksctl create nodegroup --cluster=my-cluster \
  --name=gpu-nodes \
  --node-type=g4dn.xlarge \
  --nodes=2 \
  --node-ami=auto \
  --ssh-access \
  --ssh-public-key=my-key

# GKE with GPU nodes  
gcloud container node-pools create gpu-pool \
  --cluster=my-cluster \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --num-nodes=2

# AKS with GPU nodes
az aks nodepool add \
  --resource-group myResourceGroup \
  --cluster-name myAKSCluster \
  --name gpu \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3
```

### For Kind (Development Only):

```bash
# Install NVIDIA Container Toolkit on host
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "⚠️  Kind GPU support is limited - use cloud clusters for production"
```

## Step 5: Install NVIDIA Device Plugin

```bash
# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Wait for device plugin to be ready
kubectl -n kube-system rollout status daemonset/nvidia-device-plugin-daemonset

# Verify GPU resources are available
kubectl get nodes -o custom-columns="NAME:.metadata.name,GPU:.status.capacity.nvidia\.com/gpu"
```

## Step 6: Update Training Script ConfigMap

```bash
# Update ConfigMap with GPU script
kubectl create configmap pytorch-training-script \
    --from-file=distributed_gpu_training.py=scripts/distributed_gpu_training.py \
    --dry-run=client -o yaml | kubectl apply -f -

echo "✅ GPU training script ConfigMap updated"
```

## Step 7: Submit GPU Training Job

```bash
# Submit GPU training job
kubectl apply -f configs/pytorch-gpu-job.yaml

# Check status
kubectl get pytorchjob pytorch-gpu-distributed

# Monitor GPU usage
kubectl top pods | grep pytorch-gpu

# View detailed logs
kubectl logs -l job-name=pytorch-gpu-distributed,replica-type=master -f
```

---

## 🔧 GPU Performance Tuning

### Batch Size Optimization

```bash
# Test different batch sizes
for batch_size in 64 128 256 512; do
  kubectl patch pytorchjob pytorch-gpu-distributed --patch "{
    \"spec\": {
      \"pytorchReplicaSpecs\": {
        \"Master\": {
          \"template\": {
            \"spec\": {
              \"containers\": [{
                \"name\": \"pytorch\",
                \"env\": [{
                  \"name\": \"BATCH_SIZE\",
                  \"value\": \"$batch_size\"
                }]
              }]
            }
          }
        }
      }
    }
  }"
  
  echo "Testing batch size: $batch_size"
  # Monitor performance and adjust
done
```

### Mixed Precision Training

```bash
# Enable/disable mixed precision
kubectl patch pytorchjob pytorch-gpu-distributed --patch '{
  "spec": {
    "pytorchReplicaSpecs": {
      "Master": {
        "template": {
          "spec": {
            "containers": [{
              "name": "pytorch",
              "env": [{
                "name": "MIXED_PRECISION",
                "value": "true"
              }]
            }]
          }
        }
      }
    }
  }
}'
```

## 🔍 GPU Monitoring

```bash
# Check GPU utilization in pods
kubectl exec -it $(kubectl get pods -l job-name=pytorch-gpu-distributed,replica-type=master -o jsonpath='{.items[0].metadata.name}') -- nvidia-smi

# Monitor GPU metrics
kubectl top nodes
kubectl describe node <gpu-node-name>

# Check GPU events
kubectl get events --field-selector involvedObject.kind=Pod | grep gpu
```

## 🐛 Common GPU Issues

### Issue: "CUDA out of memory"
```bash
# Reduce batch size
kubectl patch pytorchjob pytorch-gpu-distributed --patch '{
  "spec": {
    "pytorchReplicaSpecs": {
      "Master": {
        "template": {
          "spec": {
            "containers": [{
              "name": "pytorch",
              "env": [{
                "name": "BATCH_SIZE",
                "value": "64"
              }]
            }]
          }
        }
      }
    }
  }
}'

# Or increase GPU memory limits
kubectl patch pytorchjob pytorch-gpu-distributed --patch '{
  "spec": {
    "pytorchReplicaSpecs": {
      "Master": {
        "template": {
          "spec": {
            "containers": [{
              "name": "pytorch",
              "resources": {
                "limits": {
                  "nvidia.com/gpu": "2"
                }
              }
            }]
          }
        }
      }
    }
  }
}'
```

### Issue: "No GPU resources available"
```bash
# Check node GPU capacity
kubectl describe nodes | grep -A 5 "Allocatable:" | grep nvidia

# Check if device plugin is running
kubectl get pods -n kube-system | grep nvidia-device-plugin

# Restart device plugin if needed
kubectl delete pods -n kube-system -l name=nvidia-device-plugin-ds
```

## 🧹 Cleanup

```bash
# Delete GPU training job
kubectl delete pytorchjob pytorch-gpu-distributed

# Clean up GPU resources
kubectl delete -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml

# Clean up models
docker exec kubeflow-trainer-single-worker-control-plane rm -rf /tmp/output/
```

## 📚 Next Steps

After completing this example:

1. **[05-debugging](../05-debugging/)** - Learn debugging techniques
2. **[03-custom-dataset](../03-custom-dataset/)** - Train with your own data
3. **[06-common-issues](../06-common-issues/)** - Troubleshooting guide

---

**⚡ Success!** You've accelerated distributed training with GPUs. Performance should be significantly faster than CPU-only training! 