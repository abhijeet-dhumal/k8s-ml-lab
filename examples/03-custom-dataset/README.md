# 🤖 Example 07: Custom Dataset

**Goal:** Replace MNIST with your own dataset for distributed training.

**Prerequisites:**
- Working cluster (from examples 01 or 02)
- Training operator installed
- Your dataset files

**What you'll learn:**
- Dataset preparation for distributed training
- ConfigMap customization
- Storage mounting strategies
- Data loading optimization

**Estimated time:** 20 minutes

---

## Overview

This example shows how to use your own dataset instead of MNIST. We'll cover different dataset types and storage strategies.

## Dataset Types Supported

| Dataset Type | Storage Method | Best For |
|-------------|----------------|----------|
| **Small datasets** (<1GB) | ConfigMap/Volumes | Development |
| **Medium datasets** (1-10GB) | PersistentVolume | Production |
| **Large datasets** (>10GB) | Cloud storage (S3/GCS) | Scale |

## Step 1: Prepare Your Dataset

### Option A: Small Dataset (CIFAR-10 example)

```bash
# Create dataset preparation script
cat > prepare-cifar10.py << 'EOF'
import torch
import torchvision
import torchvision.transforms as transforms
import os

def prepare_mnist():
    os.makedirs('./input/MNIST', exist_ok=True)
    
    # Download MNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.MNIST(
        root='./input/MNIST', 
        train=True,
        download=True, 
        transform=transform
    )
    
    testset = torchvision.datasets.MNIST(
        root='./input/MNIST', 
        train=False,
        download=True, 
        transform=transform
    )
    
    print(f"✅ MNIST downloaded: {len(trainset)} train, {len(testset)} test")
    
    # Save dataset info
    dataset_info = {
        'name': 'MNIST',
        'classes': 10,
        'channels': 1,
        'width': 28,
        'height': 28,
        'train_samples': len(trainset),
        'test_samples': len(testset)
    }
    
    torch.save(dataset_info, './input/MNIST/dataset_info.pt')
    print("✅ Dataset info saved")

if __name__ == "__main__":
    prepare_mnist()
EOF

# Run preparation
python3 prepare-mnist.py
```

### Option B: Your Own Dataset

```bash
# Create your dataset directory structure
mkdir -p input/custom_dataset/{train,test}

# Example structure:
# input/custom_dataset/
# ├── train/
# │   ├── class1/
# │   │   ├── img1.jpg
# │   │   └── img2.jpg
# │   └── class2/
# │       ├── img3.jpg
# │       └── img4.jpg
# └── test/
#     ├── class1/
#     └── class2/

echo "Place your dataset files in input/custom_dataset/ following the structure above"
```

## Step 2: Create Custom Training Script

```bash
# Create custom training script
cat > scripts/distributed_custom_training.py << 'EOF'
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

class SimpleNet(nn.Module):
    """Simple CNN for custom datasets"""
    def __init__(self, num_classes=10, channels=1):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 16, 5, 1)
        self.conv2 = nn.Conv2d(16, 32, 5, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512, 64)  # For 28x28 MNIST images
        self.fc2 = nn.Linear(64, num_classes)
        
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

def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend='gloo')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"SUCCESS: gloo backend initialized - Rank {rank}, World size {world_size}")
    return rank, world_size

def load_dataset(rank, dataset_type='mnist'):
    """Load dataset based on type"""
    if dataset_type == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='/input/MNIST', 
            train=True, 
            transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='/input/MNIST', 
            train=False, 
            transform=transform
        )
        
        return train_dataset, test_dataset, 10, 1
    
    elif dataset_type == 'custom':
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.ImageFolder(
            root='/input/custom_dataset/train',
            transform=transform
        )
        test_dataset = torchvision.datasets.ImageFolder(
            root='/input/custom_dataset/test',
            transform=transform
        )
        
        num_classes = len(train_dataset.classes)
        print(f"Custom dataset classes: {train_dataset.classes}")
        
        return train_dataset, test_dataset, num_classes, 1
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def main():
    # Get dataset type from environment
    dataset_type = os.environ.get('DATASET_TYPE', 'mnist')
    epochs = int(os.environ.get('EPOCHS', '5'))
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    # Load dataset
    train_dataset, test_dataset, num_classes, channels = load_dataset(rank, dataset_type)
    
    if rank == 0:
        print(f"Rank {rank}: Using {dataset_type} dataset")
        print(f"Rank {rank}: {len(train_dataset)} train, {len(test_dataset)} test samples")
        print(f"Rank {rank}: {num_classes} classes, {channels} channels")
    
    # Create data loaders
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        sampler=train_sampler,
        num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    
    # Create model (same architecture as main training script)
    model = SimpleNet(num_classes=num_classes, channels=channels)
    
    # Setup DDP
    model = DDP(model)
    if rank == 0:
        print(f"Rank {rank}: DDP model created successfully")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        train_sampler.set_epoch(epoch)
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if rank == 0 and batch_idx % 20 == 0:
                accuracy = 100. * correct / total
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}: "
                      f"Loss: {loss.item():.6f}, Accuracy: {accuracy:.2f}%")
    
    # Final training accuracy
    if rank == 0:
        train_accuracy = 100. * correct / total
        print(f"Rank {rank}: Final Training Accuracy: {train_accuracy:.2f}%")
    
    # Test evaluation
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    if rank == 0:
        test_accuracy = 100. * correct / total
        print(f"Rank {rank}: Test Accuracy: {test_accuracy:.2f}%")
        
        # Save model
        os.makedirs('/app/models', exist_ok=True)
        torch.save(model.state_dict(), f'/output/trained-model.pth')
        
        # Save metadata
        metadata = f"""Dataset: {dataset_type}
Classes: {num_classes}
Channels: {channels}
Training Accuracy: {train_accuracy:.2f}%
Test Accuracy: {test_accuracy:.2f}%
Epochs: {epochs}
World Size: {world_size}
"""
        with open(f'/output/training_metadata.txt', 'w') as f:
            f.write(metadata)
        
        print("✅ Model saved successfully!")

if __name__ == "__main__":
    main()
EOF
```

## Step 3: Update ConfigMap

```bash
# Update training script ConfigMap
kubectl create configmap pytorch-training-script \
    --from-file=distributed_custom_training.py=scripts/distributed_custom_training.py \
    --dry-run=client -o yaml | kubectl apply -f -

echo "✅ Training script ConfigMap updated"
```

## Step 4: Create Custom Job Configuration

```bash
# Create job config for CIFAR-10
cat > configs/pytorch-cifar10-job.yaml << 'EOF'
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-cifar10-distributed
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
            command:
            - python3
            - /app/distributed_custom_training.py
            env:
            - name: DATASET_TYPE
              value: "cifar10"
            - name: EPOCHS
              value: "10"
            resources:
              requests:
                cpu: 800m
                memory: 1200Mi
              limits:
                cpu: 1200m
                memory: 1800Mi
            volumeMounts:
            - name: training-script
              mountPath: /app
            - name: dataset-volume
              mountPath: /data
            - name: model-volume
              mountPath: /app/models
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
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
            command:
            - python3
            - /app/distributed_custom_training.py
            env:
            - name: DATASET_TYPE
              value: "cifar10"
            - name: EPOCHS
              value: "10"
            resources:
              requests:
                cpu: 800m
                memory: 1200Mi
              limits:
                cpu: 1200m
                memory: 1800Mi
            volumeMounts:
            - name: training-script
              mountPath: /app
            - name: dataset-volume
              mountPath: /data
            - name: model-volume
              mountPath: /app/models
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
EOF
```

## Step 5: Copy Dataset to Cluster

```bash
# For Kind clusters, copy data to host mount
docker exec -it kubeflow-trainer-single-worker-control-plane mkdir -p /tmp/input
docker cp input/MNIST/. kubeflow-trainer-single-worker-control-plane:/tmp/input/MNIST/

# For other clusters, use appropriate method:
# kubectl cp input/ pod-name:/input/
# Or use persistent volumes/cloud storage
```

## Step 6: Submit Custom Training Job

```bash
# Submit CIFAR-10 training job
kubectl apply -f configs/pytorch-cifar10-job.yaml

# Check status
kubectl get pytorchjob pytorch-cifar10-distributed

# View logs
kubectl logs -l job-name=pytorch-cifar10-distributed,replica-type=master -f
```

---

## 🎯 Advanced Data Strategies

### Large Dataset with S3

```yaml
# Add S3 credentials and mount
env:
- name: AWS_ACCESS_KEY_ID
  valueFrom:
    secretKeyRef:
      name: aws-credentials
      key: access-key-id
- name: AWS_SECRET_ACCESS_KEY
  valueFrom:
    secretKeyRef:
      name: aws-credentials
      key: secret-access-key
- name: S3_BUCKET
  value: "my-training-data"
- name: DATASET_TYPE
  value: "s3"
```

### Persistent Volume for Medium Datasets

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: dataset-pvc
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 20Gi
---
# Use in job spec:
volumes:
- name: dataset-volume
  persistentVolumeClaim:
    claimName: dataset-pvc
```

## 🔍 Verification Commands

```bash
# Check dataset was loaded correctly
kubectl logs -l job-name=pytorch-cifar10-distributed,replica-type=master | grep "dataset"

# Check training progress
kubectl logs -l job-name=pytorch-cifar10-distributed,replica-type=master | grep "Accuracy"

# Check model was saved
kubectl exec -it $(kubectl get pods -l job-name=pytorch-cifar10-distributed,replica-type=master -o jsonpath='{.items[0].metadata.name}') -- ls -la /output/
```

## 🐛 Common Issues

### Issue: "Dataset not found"
```bash
# Check mount paths
kubectl describe pod $(kubectl get pods -l job-name=pytorch-cifar10-distributed,replica-type=master -o jsonpath='{.items[0].metadata.name}')

# Verify data copy
docker exec kubeflow-trainer-single-worker-control-plane ls -la /tmp/input/
```

### Issue: "Out of memory"
```bash
# Reduce batch size in training script
# Or increase memory limits in job config
kubectl patch pytorchjob pytorch-cifar10-distributed --patch '
spec:
  pytorchReplicaSpecs:
    Master:
      template:
        spec:
          containers:
          - name: pytorch
            resources:
              limits:
                memory: "4Gi"'
```

## 🧹 Cleanup

```bash
# Delete custom training job
kubectl delete pytorchjob pytorch-cifar10-distributed

# Clean up dataset files
docker exec kubeflow-trainer-single-worker-control-plane rm -rf /tmp/input/MNIST
```

## 📚 Next Steps

After completing this example:

1. **[04-gpu-training](../04-gpu-training/)** - Add GPU acceleration for larger datasets
2. **[05-debugging](../05-debugging/)** - Learn debugging techniques
3. **[02-existing-cluster](../02-existing-cluster/)** - Use with existing clusters
4. **[06-common-issues](../06-common-issues/)** - Troubleshooting guide

---

**🎉 Success!** You've successfully trained with custom datasets. Your distributed training setup can now handle any dataset type! 