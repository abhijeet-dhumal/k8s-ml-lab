# 🐛 Example 17: Debugging

**Goal:** Debug common distributed training issues with systematic troubleshooting approaches.

**Prerequisites:**
- Working distributed training setup
- Basic understanding of PyTorch distributed training
- Access to cluster logs and metrics

**What you'll learn:**
- Systematic debugging methodology
- Common distributed training issues
- Debugging tools and techniques
- Performance bottleneck identification

**Estimated time:** 45 minutes

---

## Overview

This example provides comprehensive debugging strategies for distributed PyTorch training, covering everything from setup issues to performance problems.

## Debugging Categories

| Category | Common Issues | Debug Tools |
|----------|---------------|-------------|
| **Setup** | Network, process groups | Logs, connectivity tests |
| **Performance** | Slow training, bottlenecks | Profiling, metrics |
| **Memory** | OOM, leaks | Memory monitoring |
| **Synchronization** | Deadlocks, timeouts | Process tracing |

## Step 1: Create Debug-Enabled Training Script

```bash
# Create enhanced training script with debugging
cat > scripts/distributed_training_debug.py << 'EOF'
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
import logging
import traceback
import psutil
import sys
from datetime import datetime

# Setup comprehensive logging
def setup_logging():
    """Setup detailed logging for debugging"""
    rank = int(os.environ.get('RANK', 0))
    log_format = f'[RANK {rank}] %(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'/tmp/debug_rank_{rank}.log')
        ]
    )
    
    # Set torch distributed debug
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    return logging.getLogger(__name__)

def log_system_info(logger):
    """Log comprehensive system information"""
    rank = int(os.environ.get('RANK', 0))
    
    logger.info("=== SYSTEM INFORMATION ===")
    logger.info(f"Rank: {rank}")
    logger.info(f"World Size: {os.environ.get('WORLD_SIZE', 'Unknown')}")
    logger.info(f"Master Addr: {os.environ.get('MASTER_ADDR', 'Unknown')}")
    logger.info(f"Master Port: {os.environ.get('MASTER_PORT', 'Unknown')}")
    logger.info(f"PyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {gpu_props.name}, Memory: {gpu_props.total_memory / 1e9:.1f}GB")
    
    # System resources
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(f"Memory: {psutil.virtual_memory().total / 1e9:.1f}GB")
    logger.info(f"Python Version: {sys.version}")
    
    # Environment variables
    logger.info("=== ENVIRONMENT VARIABLES ===")
    for key, value in sorted(os.environ.items()):
        if any(x in key.upper() for x in ['CUDA', 'NCCL', 'TORCH', 'RANK', 'WORLD', 'MASTER']):
            logger.info(f"{key}: {value}")

def test_connectivity(logger):
    """Test network connectivity between nodes"""
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    master_addr = os.environ.get('MASTER_ADDR', 'localhost')
    master_port = os.environ.get('MASTER_PORT', '12355')
    
    logger.info("=== CONNECTIVITY TESTS ===")
    
    # Test master connectivity
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((master_addr, int(master_port)))
        sock.close()
        
        if result == 0:
            logger.info(f"✅ Can connect to master {master_addr}:{master_port}")
        else:
            logger.error(f"❌ Cannot connect to master {master_addr}:{master_port}")
    except Exception as e:
        logger.error(f"❌ Connectivity test failed: {e}")

def debug_distributed_setup(logger):
    """Debug distributed setup with detailed logging"""
    logger.info("=== DISTRIBUTED SETUP DEBUG ===")
    
    try:
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        logger.info(f"Initializing process group - Rank: {rank}, World Size: {world_size}")
        
        # Set device before init_process_group
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            logger.info(f"Set CUDA device to {local_rank}")
            backend = 'nccl'
        else:
            backend = 'gloo'
        
        logger.info(f"Using backend: {backend}")
        
        # Initialize with timeout
        init_start = time.time()
        dist.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=300)  # 5 minute timeout
        )
        init_time = time.time() - init_start
        
        logger.info(f"✅ Process group initialized in {init_time:.2f}s")
        logger.info(f"Distributed rank: {dist.get_rank()}")
        logger.info(f"Distributed world size: {dist.get_world_size()}")
        
        # Test communication
        if world_size > 1:
            test_tensor = torch.tensor([rank], dtype=torch.float32)
            if torch.cuda.is_available():
                test_tensor = test_tensor.cuda()
            
            logger.info("Testing all_reduce communication...")
            dist.all_reduce(test_tensor)
            logger.info(f"✅ All_reduce result: {test_tensor.item()}")
            
            # Expected sum: 0 + 1 + 2 + ... + (world_size-1)
            expected = sum(range(world_size))
            if abs(test_tensor.item() - expected) < 1e-6:
                logger.info("✅ Distributed communication working correctly")
            else:
                logger.error(f"❌ Communication test failed. Expected {expected}, got {test_tensor.item()}")
        
        return rank, world_size, local_rank
        
    except Exception as e:
        logger.error(f"❌ Distributed setup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

class DebugNet(nn.Module):
    """Neural network with debugging hooks"""
    def __init__(self, num_classes=10):
        super(DebugNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
        # Debug counters
        self.forward_count = 0
        self.backward_count = 0
        
        # Register hooks for debugging
        self.register_debug_hooks()
    
    def register_debug_hooks(self):
        """Register hooks to monitor gradients and activations"""
        def forward_hook(module, input, output):
            if hasattr(output, 'register_hook'):
                def backward_hook(grad):
                    self.backward_count += 1
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        logging.error(f"NaN/Inf detected in gradients for {module.__class__.__name__}")
                output.register_hook(backward_hook)
        
        def gradient_hook(module, grad_input, grad_output):
            for i, grad in enumerate(grad_output):
                if grad is not None:
                    if torch.isnan(grad).any() or torch.isinf(grad).any():
                        logging.error(f"NaN/Inf in gradient output {i} for {module.__class__.__name__}")
        
        # Register hooks on all modules
        for name, module in self.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(gradient_hook)
    
    def forward(self, x):
        self.forward_count += 1
        
        # Check input
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("NaN/Inf detected in input")
        
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Check output
        if torch.isnan(x).any() or torch.isinf(x).any():
            logging.error("NaN/Inf detected in output")
        
        return x

def load_dataset_with_debug(logger):
    """Load dataset with debugging information"""
    logger.info("=== DATASET LOADING DEBUG ===")
    
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        train_dataset = torchvision.datasets.MNIST(
            root='/data', train=True, download=True, transform=transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root='/data', train=False, download=True, transform=transform
        )
        
        logger.info(f"✅ Dataset loaded - Train: {len(train_dataset)}, Test: {len(test_dataset)}")
        
        # Test data loading
        sample_data, sample_label = train_dataset[0]
        logger.info(f"Sample data shape: {sample_data.shape}, label: {sample_label}")
        logger.info(f"Data range: [{sample_data.min():.3f}, {sample_data.max():.3f}]")
        
        return train_dataset, test_dataset
        
    except Exception as e:
        logger.error(f"❌ Dataset loading failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def create_debug_data_loader(dataset, batch_size, rank, world_size, logger):
    """Create data loader with debugging"""
    logger.info("=== DATA LOADER DEBUG ===")
    
    try:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            drop_last=True  # Ensure consistent batch sizes
        )
        
        logger.info(f"✅ DataLoader created - Batches per epoch: {len(loader)}")
        logger.info(f"Sampler total size: {sampler.total_size}")
        logger.info(f"Rank {rank} gets {len(sampler)} samples")
        
        # Test first batch
        first_batch = next(iter(loader))
        logger.info(f"First batch shapes: {[x.shape for x in first_batch]}")
        
        return loader, sampler
        
    except Exception as e:
        logger.error(f"❌ DataLoader creation failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def monitor_memory_usage(logger, device):
    """Monitor memory usage"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9
        logger.debug(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB")
    
    memory = psutil.virtual_memory()
    logger.debug(f"System Memory - Used: {memory.used / 1e9:.2f}GB / {memory.total / 1e9:.2f}GB ({memory.percent:.1f}%)")

def main():
    # Setup logging first
    logger = setup_logging()
    logger.info("=== STARTING DISTRIBUTED TRAINING DEBUG ===")
    
    try:
        # Log system information
        log_system_info(logger)
        
        # Test connectivity
        test_connectivity(logger)
        
        # Setup distributed training
        rank, world_size, local_rank = debug_distributed_setup(logger)
        
        # Set device
        device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load dataset
        train_dataset, test_dataset = load_dataset_with_debug(logger)
        
        # Create data loaders
        batch_size = int(os.environ.get('BATCH_SIZE', '64'))
        train_loader, train_sampler = create_debug_data_loader(
            train_dataset, batch_size, rank, world_size, logger
        )
        test_loader, _ = create_debug_data_loader(
            test_dataset, batch_size, rank, world_size, logger
        )
        
        # Create model
        logger.info("=== MODEL CREATION DEBUG ===")
        model = DebugNet().to(device)
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Setup DDP
        if torch.cuda.is_available():
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        else:
            model = DDP(model)
        logger.info("✅ DDP model created")
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop with debugging
        epochs = int(os.environ.get('EPOCHS', '2'))
        logger.info(f"=== TRAINING LOOP DEBUG - {epochs} epochs ===")
        
        for epoch in range(epochs):
            train_sampler.set_epoch(epoch)
            model.train()
            
            epoch_start = time.time()
            total_loss = 0
            batch_count = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                batch_start = time.time()
                
                # Monitor memory before batch
                monitor_memory_usage(logger, device)
                
                data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
                
                # Check for NaN/Inf in input
                if torch.isnan(data).any() or torch.isinf(data).any():
                    logger.error(f"NaN/Inf detected in input data at batch {batch_idx}")
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(data)
                loss = criterion(output, target)
                
                # Check loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"NaN/Inf loss detected at batch {batch_idx}: {loss}")
                    raise ValueError("Training failed due to NaN/Inf loss")
                
                # Backward pass
                loss.backward()
                
                # Check gradients
                total_norm = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            logger.error(f"NaN/Inf gradient in {name}")
                
                total_norm = total_norm ** (1. / 2)
                
                # Gradient clipping if needed
                if total_norm > 1.0:
                    logger.warning(f"Large gradient norm detected: {total_norm:.4f}")
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                batch_time = time.time() - batch_start
                
                if batch_idx % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: "
                              f"Loss: {loss.item():.6f}, "
                              f"Grad Norm: {total_norm:.4f}, "
                              f"Batch Time: {batch_time:.3f}s")
                
                # Check for memory leaks
                if batch_idx % 50 == 0:
                    monitor_memory_usage(logger, device)
            
            epoch_time = time.time() - epoch_start
            avg_loss = total_loss / batch_count
            logger.info(f"Epoch {epoch} completed - Avg Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
            
            # Model debug info
            logger.info(f"Model forward calls: {model.module.forward_count}")
            logger.info(f"Model backward calls: {model.module.backward_count}")
        
        logger.info("=== TRAINING COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        
        # Additional debug info on failure
        logger.error("=== DEBUG INFO ON FAILURE ===")
        logger.error(f"Rank: {os.environ.get('RANK', 'Unknown')}")
        logger.error(f"World Size: {os.environ.get('WORLD_SIZE', 'Unknown')}")
        
        if torch.cuda.is_available():
            try:
                logger.error(f"CUDA Memory: {torch.cuda.memory_summary()}")
            except:
                pass
        
        # Re-raise for proper exit code
        raise

if __name__ == "__main__":
    main()
EOF
```

## Step 2: Create Debug Job Configuration

```bash
# Create debug job with enhanced logging
cat > configs/pytorch-debug-job.yaml << 'EOF'
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: pytorch-debug-distributed
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: Never  # Don't restart on failure for debugging
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
            command:
            - python3
            - /app/distributed_training_debug.py
            env:
            - name: EPOCHS
              value: "2"
            - name: BATCH_SIZE
              value: "32"
            - name: TORCH_DISTRIBUTED_DEBUG
              value: "DETAIL"
            - name: NCCL_DEBUG
              value: "INFO"
            - name: PYTHONUNBUFFERED
              value: "1"
            resources:
              requests:
                cpu: 1
                memory: 2Gi
              limits:
                cpu: 2
                memory: 4Gi
            volumeMounts:
            - name: training-script
              mountPath: /app
            - name: debug-logs
              mountPath: /tmp
            - name: data-volume
              mountPath: /data
          volumes:
          - name: training-script
            configMap:
              name: pytorch-training-script
          - name: debug-logs
            hostPath:
              path: /tmp/debug-logs
              type: DirectoryOrCreate
          - name: data-volume
            hostPath:
              path: /tmp/data
              type: DirectoryOrCreate
    Worker:
      replicas: 1
      restartPolicy: Never
      template:
        spec:
          containers:
          - name: pytorch
            image: pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
            command:
            - python3
            - /app/distributed_training_debug.py
            env:
            - name: EPOCHS
              value: "2"
            - name: BATCH_SIZE
              value: "32"
            - name: TORCH_DISTRIBUTED_DEBUG
              value: "DETAIL"
            - name: NCCL_DEBUG
              value: "INFO"
            - name: PYTHONUNBUFFERED
              value: "1"
            resources:
              requests:
                cpu: 1
                memory: 2Gi
              limits:
                cpu: 2
                memory: 4Gi
            volumeMounts:
            - name: training-script
              mountPath: /app
            - name: debug-logs
              mountPath: /tmp
            - name: data-volume
              mountPath: /data
          volumes:
          - name: training-script
            configMap:
              name: pytorch-training-script
          - name: debug-logs
            hostPath:
              path: /tmp/debug-logs
              type: DirectoryOrCreate
          - name: data-volume
            hostPath:
              path: /tmp/data
              type: DirectoryOrCreate
EOF
```

## Step 3: Create Debugging Utilities

```bash
# Create debugging utility scripts
cat > scripts/debug_utilities.py << 'EOF'
#!/usr/bin/env python3
"""
Debugging utilities for distributed PyTorch training
"""

import subprocess
import json
import yaml
import sys
import time
import requests

def check_cluster_health():
    """Check overall cluster health"""
    print("=== CLUSTER HEALTH CHECK ===")
    
    # Check nodes
    result = subprocess.run(['kubectl', 'get', 'nodes', '-o', 'json'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        nodes = json.loads(result.stdout)
        print(f"✅ Found {len(nodes['items'])} nodes")
        
        for node in nodes['items']:
            name = node['metadata']['name']
            status = 'Ready' if any(
                c['type'] == 'Ready' and c['status'] == 'True' 
                for c in node['status']['conditions']
            ) else 'Not Ready'
            print(f"  Node {name}: {status}")
    else:
        print(f"❌ Failed to get nodes: {result.stderr}")

def check_training_operator():
    """Check training operator status"""
    print("\n=== TRAINING OPERATOR CHECK ===")
    
    result = subprocess.run([
        'kubectl', 'get', 'deployment', 'training-operator', 
        '-n', 'kubeflow', '-o', 'json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        deployment = json.loads(result.stdout)
        replicas = deployment['status']['replicas']
        ready_replicas = deployment['status'].get('readyReplicas', 0)
        print(f"✅ Training operator: {ready_replicas}/{replicas} replicas ready")
    else:
        print(f"❌ Training operator not found or error: {result.stderr}")

def analyze_job_logs(job_name):
    """Analyze logs from a PyTorch job"""
    print(f"\n=== ANALYZING LOGS FOR {job_name} ===")
    
    # Get pods for the job
    result = subprocess.run([
        'kubectl', 'get', 'pods', '-l', f'job-name={job_name}', '-o', 'json'
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Failed to get pods: {result.stderr}")
        return
    
    pods = json.loads(result.stdout)
    
    for pod in pods['items']:
        pod_name = pod['metadata']['name']
        replica_type = pod['metadata']['labels'].get('replica-type', 'unknown')
        phase = pod['status']['phase']
        
        print(f"\n--- Pod: {pod_name} ({replica_type}) - {phase} ---")
        
        # Get pod logs
        log_result = subprocess.run([
            'kubectl', 'logs', pod_name, '--tail=50'
        ], capture_output=True, text=True)
        
        if log_result.returncode == 0:
            logs = log_result.stdout
            
            # Analyze for common issues
            if 'ERROR' in logs:
                print("🔍 Found ERROR messages:")
                for line in logs.split('\n'):
                    if 'ERROR' in line:
                        print(f"  {line}")
            
            if 'timeout' in logs.lower():
                print("⏰ Found timeout issues:")
                for line in logs.split('\n'):
                    if 'timeout' in line.lower():
                        print(f"  {line}")
            
            if 'connection' in logs.lower() and 'refused' in logs.lower():
                print("🔌 Found connection issues:")
                for line in logs.split('\n'):
                    if 'connection' in line.lower() and 'refused' in line.lower():
                        print(f"  {line}")
        else:
            print(f"❌ Failed to get logs: {log_result.stderr}")

def debug_networking():
    """Debug networking issues"""
    print("\n=== NETWORKING DEBUG ===")
    
    # Check services
    result = subprocess.run([
        'kubectl', 'get', 'services', '-o', 'json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        services = json.loads(result.stdout)
        print(f"✅ Found {len(services['items'])} services")
        
        for service in services['items']:
            name = service['metadata']['name']
            cluster_ip = service['spec'].get('clusterIP', 'None')
            ports = service['spec'].get('ports', [])
            print(f"  Service {name}: {cluster_ip}, Ports: {[p['port'] for p in ports]}")
    
    # Check network policies
    result = subprocess.run([
        'kubectl', 'get', 'networkpolicies', '-o', 'json'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        policies = json.loads(result.stdout)
        print(f"Network policies: {len(policies['items'])}")

def check_resource_constraints():
    """Check resource constraints and usage"""
    print("\n=== RESOURCE CONSTRAINTS CHECK ===")
    
    # Check node resources
    result = subprocess.run([
        'kubectl', 'top', 'nodes'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Node resource usage:")
        print(result.stdout)
    else:
        print("❌ Could not get node resource usage")
    
    # Check pod resources
    result = subprocess.run([
        'kubectl', 'top', 'pods'
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Pod resource usage:")
        print(result.stdout)
    else:
        print("❌ Could not get pod resource usage")

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_utilities.py <command> [args]")
        print("Commands:")
        print("  health - Check cluster health")
        print("  operator - Check training operator")
        print("  logs <job_name> - Analyze job logs")
        print("  network - Debug networking")
        print("  resources - Check resource constraints")
        print("  all <job_name> - Run all checks")
        return
    
    command = sys.argv[1]
    
    if command == 'health':
        check_cluster_health()
    elif command == 'operator':
        check_training_operator()
    elif command == 'logs' and len(sys.argv) > 2:
        analyze_job_logs(sys.argv[2])
    elif command == 'network':
        debug_networking()
    elif command == 'resources':
        check_resource_constraints()
    elif command == 'all' and len(sys.argv) > 2:
        check_cluster_health()
        check_training_operator()
        debug_networking()
        check_resource_constraints()
        analyze_job_logs(sys.argv[2])
    else:
        print(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
EOF

chmod +x scripts/debug_utilities.py
```

## Step 4: Create Debug Makefile Targets

```bash
# Add debug targets to Makefile
cat >> Makefile << 'EOF'

# Debugging targets
.PHONY: debug-job debug-logs debug-cluster debug-all debug-cleanup

debug-job: ## Submit debug training job
	@echo "Submitting debug training job..."
	kubectl create configmap pytorch-training-script \
		--from-file=distributed_training_debug.py=scripts/distributed_training_debug.py \
		--dry-run=client -o yaml | kubectl apply -f -
	kubectl apply -f configs/pytorch-debug-job.yaml
	@echo "Debug job submitted. Monitor with: make debug-logs"

debug-logs: ## Watch debug job logs
	@echo "Watching debug job logs..."
	kubectl logs -l job-name=pytorch-debug-distributed,replica-type=master -f

debug-cluster: ## Run cluster health checks
	@echo "Running cluster health checks..."
	python3 scripts/debug_utilities.py health
	python3 scripts/debug_utilities.py operator
	python3 scripts/debug_utilities.py network
	python3 scripts/debug_utilities.py resources

debug-analyze: ## Analyze debug job logs
	@echo "Analyzing debug job logs..."
	python3 scripts/debug_utilities.py logs pytorch-debug-distributed

debug-all: debug-cluster debug-job ## Run complete debug workflow
	@echo "Waiting for job to start..."
	@sleep 30
	@echo "Analyzing logs..."
	@$(MAKE) debug-analyze

debug-cleanup: ## Clean up debug resources
	@echo "Cleaning up debug resources..."
	kubectl delete pytorchjob pytorch-debug-distributed --ignore-not-found
	docker exec kubeflow-trainer-single-worker-control-plane rm -rf /tmp/debug-logs/*

EOF
```

## Step 5: Common Issue Diagnostics

### Issue: Process Group Initialization Fails

```bash
# Check master node connectivity
kubectl exec -it pytorch-debug-distributed-master-0 -- nslookup pytorch-debug-distributed-master-0

# Check environment variables
kubectl exec -it pytorch-debug-distributed-master-0 -- env | grep -E "(MASTER|RANK|WORLD)"

# Test port connectivity
kubectl exec -it pytorch-debug-distributed-master-0 -- nc -zv pytorch-debug-distributed-master-0 23456
```

### Issue: NCCL Communication Errors

```bash
# Check NCCL debug output
kubectl logs pytorch-debug-distributed-master-0 | grep NCCL

# Test GPU availability
kubectl exec -it pytorch-debug-distributed-master-0 -- nvidia-smi

# Check CUDA environment
kubectl exec -it pytorch-debug-distributed-master-0 -- python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Memory Issues

```bash
# Monitor memory usage
kubectl top pods pytorch-debug-distributed-master-0

# Check for OOM kills
kubectl describe pod pytorch-debug-distributed-master-0 | grep -A 5 "Last State"

# Check memory limits
kubectl get pod pytorch-debug-distributed-master-0 -o jsonpath='{.spec.containers[0].resources}'
```

## Step 6: Performance Debugging

```bash
# Create performance profiling script
cat > scripts/profile_training.py << 'EOF'
import torch
import torch.profiler
import os
import time
from torch.utils.data import DataLoader
import torchvision

def profile_training_step():
    """Profile a single training step"""
    
    # Simple model and data
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, 3),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 10)
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.CrossEntropyLoss()
    
    # Sample data
    batch_size = 32
    x = torch.randn(batch_size, 1, 28, 28)
    y = torch.randint(0, 10, (batch_size,))
    
    if torch.cuda.is_available():
        x, y = x.cuda(), y.cuda()
    
    # Profile training step
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('/tmp/profiler_logs'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for step in range(5):
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            prof.step()
    
    print("Profiling completed. View results with: tensorboard --logdir=/tmp/profiler_logs")
    
    # Print summary
    print("\nTop 10 operations by self CPU time:")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

if __name__ == "__main__":
    profile_training_step()
EOF
```

## Step 7: Systematic Debugging Workflow

```bash
# Create systematic debugging script
cat > scripts/systematic_debug.sh << 'EOF'
#!/bin/bash

echo "=== SYSTEMATIC DEBUGGING WORKFLOW ==="

# Step 1: Basic cluster health
echo "Step 1: Checking cluster health..."
python3 scripts/debug_utilities.py health

# Step 2: Training operator
echo "Step 2: Checking training operator..."
python3 scripts/debug_utilities.py operator

# Step 3: Submit debug job
echo "Step 3: Submitting debug job..."
make debug-job

# Step 4: Wait and monitor
echo "Step 4: Waiting for job to start (30s)..."
sleep 30

# Step 5: Analyze logs
echo "Step 5: Analyzing logs..."
python3 scripts/debug_utilities.py logs pytorch-debug-distributed

# Step 6: Check resources
echo "Step 6: Checking resource usage..."
python3 scripts/debug_utilities.py resources

# Step 7: Network check
echo "Step 7: Checking networking..."
python3 scripts/debug_utilities.py network

echo "=== DEBUG WORKFLOW COMPLETE ==="
echo "Manual steps:"
echo "1. Check job status: kubectl get pytorchjob pytorch-debug-distributed"
echo "2. Watch logs: make debug-logs"
echo "3. Describe pod: kubectl describe pod <pod-name>"
echo "4. Clean up: make debug-cleanup"
EOF

chmod +x scripts/systematic_debug.sh
```

---

## 🔍 Running Debug Session

```bash
# Run complete debug workflow
./scripts/systematic_debug.sh

# Or step by step
make debug-cluster
make debug-job
make debug-logs
make debug-analyze
```

## 🐛 Issue-Specific Debugging

### Initialization Issues
```bash
# Check environment variables
kubectl get pytorchjob pytorch-debug-distributed -o yaml | grep -A 20 env

# Test connectivity manually
kubectl exec -it pytorch-debug-distributed-master-0 -- python -c "
import socket
s = socket.socket()
s.connect(('pytorch-debug-distributed-master-0', 23456))
print('Connection successful')
"
```

### Performance Issues
```bash
# Run profiler
kubectl exec -it pytorch-debug-distributed-master-0 -- python /app/scripts/profile_training.py

# Check CPU/memory limits
kubectl top pod pytorch-debug-distributed-master-0
```

### Data Loading Issues
```bash
# Test data access
kubectl exec -it pytorch-debug-distributed-master-0 -- ls -la /input/

# Check data loader
kubectl exec -it pytorch-debug-distributed-master-0 -- python -c "
import torch
from torchvision import datasets
dataset = datasets.MNIST('/data', download=True)
print(f'Dataset size: {len(dataset)}')
"
```

## 🧹 Cleanup

```bash
# Clean up debug resources
make debug-cleanup

# Remove debug scripts
rm -f scripts/debug_utilities.py scripts/profile_training.py scripts/systematic_debug.sh
```

## 📚 Next Steps

After mastering debugging:

1. **Performance optimization** based on debug findings
2. **Automated monitoring** to prevent issues
3. **Production debugging strategies**

---

**🐛 Success!** You now have comprehensive debugging tools and methodologies for distributed PyTorch training! 