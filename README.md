# K8s ML Training Lab - Local Infrastructure Setup

🚀 **Complete infrastructure setup for Kubernetes-based ML workloads with automated local cluster provisioning and operator management**

A production-ready, infrastructure-first solution that gets you from zero to a fully configured K8s ML environment in 15 minutes.

## 🎯 What This Project Provides

✅ **Automated Cluster Setup**: Kind clusters optimized for ML workloads on laptops/workstations  
✅ **Operator Management**: Automated Kubeflow Training Operator installation and configuration  
✅ **Infrastructure as Code**: Declarative cluster configurations and resource management  
✅ **Multi-Backend Support**: Flexible infrastructure supporting various ML frameworks  
✅ **Resource Optimization**: Memory and CPU configurations tuned for local development  
✅ **Fault-Tolerant Setup**: Robust infrastructure with multiple fallback mechanisms  
✅ **Ready-to-Use Examples**: Pre-configured distributed PyTorch training as proof-of-concept  

## 📁 Project Structure

```
k8s-ml-lab/
├── bin/                                # Executable scripts
│   └── setup.sh                       # Infrastructure automation script
├── configs/                            # Infrastructure configurations
│   ├── kind-cluster-config.yaml        # Kind cluster specification
│   ├── pytorch-distributed-job.yaml    # Sample workload configuration
│   └── pytorch-test-job.yaml           # Test workload configuration
├── scripts/                            # Sample ML workloads
│   ├── mnist_training.py               # Unified training script (single + distributed)
│   └── test_mnist_model.py             # Model inference example
├── input/                              # Input datasets (auto-populated)
├── output/                             # Training outputs (auto-created)
├── examples/                           # Infrastructure examples and guides
│   ├── README.md                       # 📚 Comprehensive documentation
│   ├── 01-complete-workflow/           # Complete infrastructure + workload demo
│   ├── 02-existing-cluster/            # Existing cluster integration
│   ├── 03-custom-dataset/              # Custom workload configurations
│   ├── 04-gpu-training/                # GPU-enabled cluster setup
│   ├── 05-debugging/                   # Infrastructure debugging guide
│   └── 06-common-issues/               # Infrastructure troubleshooting
├── Makefile                           # Infrastructure automation commands
└── README.md                          # This file
```

## 🚀 Quick Infrastructure Setup (15 minutes)

### Prerequisites
- **macOS 11+** or **Linux** (Ubuntu, Fedora, Debian, etc.)
- **8GB+ RAM** (16GB recommended)
- **4+ CPU cores**, **10GB free disk space**
- **Docker or Podman** (container runtime)

### 1. Infrastructure Setup
```bash
# Clone repository
git clone https://github.com/<your-username>/k8s-ml-lab.git
cd k8s-ml-lab

# Automated infrastructure setup (recommended)
make setup                    # Complete setup: cluster + operators + training environment

# Alternative: Configure existing cluster
make use-existing             # For EKS, GKE, AKS, minikube, etc.
```

### 2. Verify Infrastructure
```bash
# Comprehensive system verification (recommended first step)
make verify-system       # Checks system requirements + all dependencies

# Check cluster status
make status

# Submit test workload
make submit-job

# View workload logs
make logs

# OR: Run complete end-to-end workflow
make run-e2e-workflow    # Runs training + inference + testing automatically
```

### 3. Test ML Workload
```bash
# Test the sample distributed training workload
python scripts/test_mnist_model.py

# OR: Use make command for easier testing
make inference                                    # Test with built-in images
TEST_IMAGE=path/to/digit.png make inference       # Test single custom image
TEST_IMAGES_DIR=my_digits/ make inference         # Test directory of images
```

## 📊 Expected Infrastructure Results

```
SUCCESS: Kind cluster 'ml-training-cluster' created
SUCCESS: Kubeflow Training Operator installed
SUCCESS: gloo backend initialized - Rank 0, World size 2
Rank 0: Using pre-downloaded MNIST dataset (60000 train, 10000 test)
✅ Infrastructure ready for ML workloads!
✅ Sample workload completed successfully!
```

**Generated Infrastructure:**
- Kubernetes cluster with ML-optimized configuration
- Kubeflow Training Operator for distributed workloads
- Persistent storage for datasets and models
- Network policies and resource quotas
- Sample workload demonstrating capabilities

## 🔄 End-to-End Workflow

The `make run-e2e-workflow` command runs the complete end-to-end workflow automation:

1. **Training Phase**: Submits distributed PyTorch training job
2. **Monitoring Phase**: Tracks job progress and collects logs
3. **Inference Phase**: Tests trained model with sample images
4. **Results Phase**: Generates performance reports and saves outputs

**What it does:**
- Creates and submits PyTorch distributed training job
- Monitors job completion and downloads training logs
- Extracts trained model from completed pods
- Runs inference tests on sample handwritten digit images
- Generates training metrics and accuracy reports
- Saves all outputs to `output/latest/` directory

**Example output:**
```
✅ Training job submitted and completed
✅ Model extracted: output/latest/trained-model.pth
✅ Inference tests passed: 8/10 correct predictions
✅ Training metrics saved: output/latest/training_metadata.txt
```

## 🔍 System Verification

The `make verify-system` command performs comprehensive verification of your system readiness:

**System Requirements Check:**
- Memory: minimum 8GB, recommended 16GB
- CPU: minimum 4 cores
- Disk space: minimum 10GB free
- Operating system and architecture detection

**Dependencies Verification:**
- Container runtime: Docker/Podman installation and status
- Python: version and availability
- kubectl: Kubernetes CLI installation and version
- kind: Kubernetes in Docker installation
- Python packages: PyTorch, torchvision, requests, PyYAML

**Infrastructure Status:**
- Kubernetes cluster accessibility
- Kubeflow Training Operator installation status
- Overall readiness assessment

**Example output:**
```
✅ Memory: 16GB (sufficient)
✅ CPU cores: 8 (sufficient)
✅ Disk space: 45GB free (sufficient)
✅ Docker: installed and running
✅ Python: 3.11.5
✅ kubectl: v1.28.0
✅ kind: v0.20.0
✅ Python dependencies: PyTorch, torchvision, requests, PyYAML installed
✅ Kubernetes cluster: accessible
✅ Kubeflow Training Operator: installed
✅ System verification completed - all dependencies are ready!
```

**Use this command:**
- Before starting any setup to identify missing dependencies
- After setup to confirm everything is working
- When troubleshooting issues
- As part of CI/CD pipeline validation

## 📚 Documentation

**👉 [Complete Documentation](examples/README.md)** - Detailed infrastructure guides, architecture, troubleshooting, and advanced configurations

### Quick Links
- **[Setup Guide](examples/README.md#setup-guide)** - Detailed installation and configuration
- **[Architecture](examples/README.md#architecture)** - Infrastructure components and design
- **[Complete Workflow](examples/01-complete-workflow/)** - End-to-end infrastructure + workload demo
- **[Existing Clusters](examples/02-existing-cluster/)** - Integrate with EKS, GKE, AKS, etc.
- **[Custom Workloads](examples/03-custom-dataset/)** - Configure your own ML workloads
- **[GPU Infrastructure](examples/04-gpu-training/)** - GPU-enabled cluster setup
- **[Debugging](examples/05-debugging/)** - Infrastructure debugging techniques
- **[Troubleshooting](examples/06-common-issues/)** - Common infrastructure problems and solutions

## 🔧 Common Infrastructure Commands

```bash
# Infrastructure Management
make setup               # Complete infrastructure setup (cluster + dependencies + training env)
make verify-system       # Comprehensive system and dependency verification
make use-existing        # Use existing cluster (skip cluster creation)

# Training & Workflows
make submit-job          # Submit PyTorch distributed training job
make run-e2e-workflow    # Run complete end-to-end workflow (training + inference + results)
make inference           # Run model inference on test images (TEST_IMAGE=path or TEST_IMAGES_DIR=path)
make status              # Show job status, pods, and recent events
make logs                # View logs from master pod (real-time)
make restart             # Restart training job (delete + submit)

# Debugging & Monitoring
make debug               # Show comprehensive debugging information

# Cleanup
make cleanup             # Clean up jobs and resources (keep cluster)
make cleanup-all         # Delete entire Kind cluster and all resources

# Aliases (for compatibility)
make check-requirements  # Alias for verify-system
make install-operator    # Install Kubeflow training operator (standalone)
```

## 🎨 Quick Infrastructure Customization

**Scale Infrastructure:**
```yaml
# configs/pytorch-distributed-job.yaml
Worker:
  replicas: 3  # Scale workers from 1 to 3
```

**Custom Cluster Configuration:**
```yaml
# configs/kind-cluster-config.yaml
nodes:
- role: control-plane
- role: worker
- role: worker  # Add more workers
```

**Configure Your Workloads:**
```python
# scripts/mnist_training.py
def load_dataset(rank):
    # Replace with your dataset
    train_dataset = YourDataset('/input/your-data')
    return train_dataset, test_dataset
```

## 🧹 Cleanup

```bash
# Delete workloads only
make cleanup

# Delete entire infrastructure (Kind cluster)
make cleanup-all
```