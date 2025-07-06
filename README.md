# Distributed PyTorch Training with Kubeflow

🚀 **Complete setup for distributed PyTorch training on local Kubernetes clusters using Kubeflow Training Operator**

A production-ready, beginner-friendly solution that gets you from zero to distributed ML training in 15 minutes.

## 🎯 What This Project Provides

✅ **Real Distributed Training**: 2+ process PyTorch training with gradient synchronization  
✅ **Local Kubernetes**: Kind cluster optimized for laptops/workstations  
✅ **Kubeflow Integration**: Industry-standard training operator  
✅ **MNIST Dataset**: 60,000 real images with 90%+ accuracy results  
✅ **Model Inference**: Test trained models with custom handwritten digit images  
✅ **Fault Tolerance**: Multiple backend fallbacks (NCCL → Gloo → Manual)  
✅ **Resource Optimized**: Efficient memory usage for M1/M2 Macs  

## 📁 Project Structure

```
distributed-pytorch-training-setup/
├── scripts/                            # Core training scripts
│   ├── distributed_mnist_training.py   # Main distributed training
│   ├── simple_single_pod_training.py   # Single-pod training
│   └── test_mnist_model.py             # Model inference and testing
├── configs/                            # Kubernetes configurations
│   ├── pytorch-distributed-job.yaml    # PyTorch training job
│   └── kind-cluster-config.yaml        # Kind cluster configuration
├── input/                              # Input datasets (auto-populated)
├── output/                             # Training outputs (auto-created)
├── examples/                           # Detailed examples and guides
│   ├── README.md                       # 📚 Comprehensive documentation
│   ├── 01-complete-workflow/           # Complete training + inference
│   │   └── test_images/                # Default test images for inference
│   ├── 02-existing-cluster/            # Using existing clusters
│   ├── 03-custom-dataset/              # Custom dataset training
│   ├── 04-gpu-training/                # GPU acceleration
│   ├── 05-debugging/                   # Debugging guide
│   └── 06-common-issues/               # Troubleshooting guide
├── test_images/                        # Optional: Your custom test images
├── setup.sh                           # Main setup script
├── Makefile                           # Automation commands
└── README.md                          # This file
```

## 🚀 Quick Start (15 minutes)

### Prerequisites
- **macOS 11+** or **Linux** (Ubuntu, Fedora, Debian, etc.)
- **8GB+ RAM** (16GB recommended)
- **4+ CPU cores**, **10GB free disk space**
- **Docker or Podman** (container runtime)

### 1. Setup
```bash
# Clone repository
git clone https://github.com/<your-username>/distributed-pytorch-training-setup.git
cd distributed-pytorch-training-setup

# Automated setup (recommended)
./setup.sh                    # Creates cluster + installs dependencies
./setup.sh setup-training     # Installs operator + prepares dataset

# Alternative: Use existing cluster
./setup.sh use-existing       # For EKS, GKE, AKS, minikube, etc.
```

### 2. Run Training
```bash
# Submit training job
make submit-job

# Check status
make status

# View logs
make logs

# Get results
make results
```

### 3. Test Model
```bash
# Test trained model with sample images
python scripts/test_mnist_model.py

# Test with your own image
python scripts/test_mnist_model.py --image path/to/digit.png
```

## 📊 Expected Results

```
SUCCESS: gloo backend initialized - Rank 0, World size 2
Rank 0: Using pre-downloaded MNIST dataset (60000 train, 10000 test)
Rank 0, Epoch 0, Batch 0: Loss: 2.298154, Accuracy: 6.25%
Rank 0, Epoch 0, Batch 20: Loss: 0.524573, Accuracy: 84.38%
...
Rank 0: Final Training Accuracy: 90.61%
Rank 0: Test Accuracy: 90.25%
✅ Model saved successfully!
```

**Generated Files:**
- `output/latest/trained-model.pth` - Trained model
- `output/latest/training_metadata.txt` - Training metrics
- `output/latest/*-pod-logs.txt` - Pod logs

## 📚 Documentation

**👉 [Complete Documentation](examples/README.md)** - Detailed guides, architecture, troubleshooting, and advanced usage

### Quick Links
- **[Setup Guide](examples/README.md#setup-guide)** - Detailed installation and configuration
- **[Architecture](examples/README.md#architecture)** - How distributed training works
- **[Complete Workflow](examples/01-complete-workflow/)** - End-to-end training + inference
- **[Existing Clusters](examples/02-existing-cluster/)** - Use EKS, GKE, AKS, etc.
- **[Custom Dataset](examples/03-custom-dataset/)** - Train with your own data
- **[GPU Training](examples/04-gpu-training/)** - Accelerate with GPUs
- **[Debugging](examples/05-debugging/)** - Advanced debugging techniques
- **[Troubleshooting](examples/06-common-issues/)** - Common problems and solutions

## 🔧 Common Commands

```bash
# Infrastructure
make setup               # Full setup (cluster + training)
make use-existing        # Use existing cluster

# Training
make submit-job          # Submit training job
make status              # Check job status
make logs                # View logs
make results             # Get training results

# Cleanup
make cleanup             # Clean up resources (keep cluster)
make delete-cluster      # Delete kind cluster (Kind only)
```

## 🎨 Quick Customization

**Scale Workers:**
```yaml
# configs/pytorch-distributed-job.yaml
Worker:
  replicas: 3  # Change from 1 to 3
```

**Use Your Dataset:**
```python
# scripts/distributed_mnist_training.py
def load_dataset(rank):
    # Replace MNIST with your dataset
    train_dataset = YourDataset('/input/your-data')
    return train_dataset, test_dataset
```

## 🧹 Cleanup

```bash
# Delete training job only
make cleanup

# Delete everything (Kind cluster only)
make cleanup-all
```

## 🤝 Contributing

1. Fork this repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the ML community**

*Transform your laptop into a distributed training powerhouse!* 🚀 