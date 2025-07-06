# 01: Complete Workflow - Training & Inference

🚀 **End-to-end distributed PyTorch training and model inference demonstration**

This example takes you through the complete machine learning workflow:
1. **Infrastructure Setup** - Create Kubernetes cluster and install dependencies
2. **Training Operator** - Deploy Kubeflow training operator
3. **Distributed Training** - Train MNIST CNN model with 2+ worker processes
4. **Model Inference** - Test trained model with real handwritten digits
5. **Batch Processing** - Analyze multiple images at once

## 🎯 What This Example Demonstrates

✅ **Complete ML Pipeline**: From raw data to deployed model  
✅ **Real Distributed Training**: Multi-process PyTorch with gradient synchronization  
✅ **Model Testing**: Real handwritten digit images  
✅ **Production Patterns**: Unique output directories, log collection, monitoring  
✅ **Automation**: Single script runs entire workflow  

## 🚀 Quick Start

### Prerequisites
- Docker or Podman installed and running
- 8GB+ RAM, 10GB+ disk space
- Internet connection for downloading images/datasets

### Option A: Full Automated Workflow (Recommended)
```bash
# Run complete training + inference workflow
cd examples/01-complete-workflow
./run-complete-workflow.sh all

# This will:
# 1. Set up infrastructure (cluster + dependencies)
# 2. Install training operator
# 3. Train MNIST model (distributed PyTorch)
# 4. Test model with handwritten digit images
# 5. Generate comprehensive results
```

### Option B: Step-by-Step Learning (Educational)
```bash
# Phase 1: Infrastructure Setup
./run-complete-workflow.sh setup

# Phase 2: Training Operator Installation
./run-complete-workflow.sh install-training-operator

# Phase 3: Distributed Training  
./run-complete-workflow.sh training

# Phase 4: Model Inference
./run-complete-workflow.sh inference

# Phase 5: Results Summary
./run-complete-workflow.sh results

# Each phase shows clear next steps and handles dependencies
```

### Option C: Individual Components
```bash
# 1. Training only
./run-training.sh

# 2. Inference only (requires trained model)
./run-inference.sh

# 3. Clean up
./cleanup.sh
```

### Getting Help
```bash
# Show all available phases and options
./run-complete-workflow.sh help

# Environment variables for customization
EPOCHS=10 WORKERS=3 ./run-complete-workflow.sh training

# Inference customization
TEST_IMAGE=my_digit.jpg ./run-complete-workflow.sh inference
TEST_IMAGES_DIR=my_images/ ./run-complete-workflow.sh inference
# Note: By default, uses examples/01-complete-workflow/test_images/ if available
MODEL_PATH=output/older_model.pth ./run-complete-workflow.sh inference
```

## 📊 Expected Results

> **📋 Full Sample Output**: See [`sample-complete-workflow-output.txt`](sample-complete-workflow-output.txt) for complete console output from running `./run-complete-workflow.sh all`

### Training Output
```
🚀 Setting up distributed PyTorch training...
✅ Kind cluster ready: pytorch-training-cluster
✅ Training operator installed
✅ MNIST dataset prepared
🎯 Submitting training job...

SUCCESS: gloo backend initialized - Rank 0, World size 2
Rank 0: Using pre-downloaded MNIST dataset (60000 train, 10000 test)
Rank 0, Epoch 0, Batch 0: Loss: 2.298154, Accuracy: 6.25%
Rank 0, Epoch 0, Batch 20: Loss: 0.524573, Accuracy: 84.38%
...
Rank 0: Final Training Accuracy: 90.61%
Rank 0: Test Accuracy: 90.25%
✅ Model saved successfully!

📁 Training artifacts saved to: output/pytorch-single-worker-distributed_2025-01-06_18-30-45/
```

### Inference Output
```
🔍 Testing with real handwritten digits...

📸 Testing digit3.jpg:
Predicted: 3 (Confidence: 99.8%)
Top predictions: 3 (99.8%), 8 (0.1%), 5 (0.1%)

📸 Testing digit4.jpg:
Predicted: 4 (Confidence: 97.2%)
Top predictions: 4 (97.2%), 9 (2.1%), 7 (0.4%)

📸 Testing digit6.jpg:
Predicted: 6 (Confidence: 98.5%)
Top predictions: 6 (98.5%), 0 (1.2%), 8 (0.2%)

✅ All real digit predictions correct!
```

### Generated Files

Each PyTorchJob creates a unique output directory to prevent conflicts:

```
output/
├── pytorch-single-worker-distributed_2025-01-06_18-30-45/  # Job-specific directory
│   ├── trained-model.pth              # Trained PyTorch model
│   ├── training_metadata.txt          # Training metrics and results
│   ├── master-pod-logs.txt            # Master process logs
│   ├── worker-pod-logs.txt            # Worker process logs
│   ├── job-info.txt                   # Job execution details
│   └── inference-results.txt          # Model testing results
├── pytorch-single-worker-distributed_2025-01-06_20-15-23/  # Another job
│   ├── trained-model.pth
│   └── ...
└── latest -> pytorch-single-worker-distributed_2025-01-06_20-15-23/  # Symlink to most recent

test_images/
├── digit3.jpg                     # Real handwritten digits
├── digit4.jpg
└── digit6.jpg
```

**Directory Naming Convention**: `{job-name}_{timestamp}`
- **Job Name**: PyTorchJob resource name (e.g., `pytorch-single-worker-distributed`)
- **Timestamp**: Execution time in `YYYY-MM-DD_HH-MM-SS` format
- **Symlink**: `output/latest/` always points to the most recent job

**Benefits of Job-Specific Directories**:
✅ **No Conflicts**: Multiple training runs won't overwrite each other  
✅ **Traceability**: Easy to identify which artifacts belong to which job  
✅ **Comparison**: Compare models from different training runs  
✅ **History**: Keep complete training history for analysis  
✅ **Rollback**: Easy to revert to previous model versions

## 🔍 Inference Modes

The inference phase supports multiple modes for flexible model testing:

### 1. Default Mode (Built-in Images)
```bash
# Uses built-in handwritten digit images
./run-complete-workflow.sh inference
```

### 2. Single Image Mode
```bash
# Test a single custom image
TEST_IMAGE=path/to/your_digit.jpg ./run-complete-workflow.sh inference
TEST_IMAGE=my_handwritten_7.png ./run-complete-workflow.sh inference
```

### 3. Directory Mode
```bash
# Test all images in a directory
TEST_IMAGES_DIR=my_test_images/ ./run-complete-workflow.sh inference
TEST_IMAGES_DIR=/path/to/digits/ ./run-complete-workflow.sh inference
```

### 4. Custom Model
```bash
# Use a specific model (works with any mode above)
MODEL_PATH=output/older_model.pth ./run-complete-workflow.sh inference

# Combine with custom images
TEST_IMAGE=my_digit.jpg MODEL_PATH=output/older_model.pth ./run-complete-workflow.sh inference
```

### Supported Image Formats
- **JPEG**: `.jpg`, `.jpeg`
- **PNG**: `.png`
- **GIF**: `.gif`
- **Any format supported by PIL/Pillow**

## 🛠️ Training Customization

### Modify Training Parameters
```bash
# Environment variables for training
EPOCHS=10 WORKERS=3 BATCH_SIZE=128 ./run-complete-workflow.sh training
```

### Use Different Model Architecture
```python
# Edit scripts/distributed_mnist_training.py
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        # Your custom architecture here
```

## 🔧 Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Infrastructure │ -> │ Training        │ -> │ Distributed     │ -> │ Model           │
│  Setup          │    │ Operator        │    │ Training        │    │ Inference       │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • Kind Cluster  │    │ • Kubeflow      │    │ • Master Pod    │    │ • Real Images   │
│ • Dependencies  │    │ • PyTorchJob    │    │ • Worker Pod(s) │    │ • Batch Process │
│ • MNIST Data    │    │ • CRD Install   │    │ • Gradient Sync │    │ • Batch Process │
│ • Storage       │    │ • Operator Pod  │    │ • Model Save    │    │ • Confidence    │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎓 Phase-Based Learning

The step-by-step approach offers several advantages:

✅ **Understanding**: See exactly what happens at each stage  
✅ **Debugging**: Isolate issues to specific phases  
✅ **Customization**: Modify parameters between phases  
✅ **Reusability**: Re-run inference without retraining  
✅ **Development**: Test changes to individual components  

### Example Learning Path
```bash
# Learn infrastructure setup
./run-complete-workflow.sh setup
kubectl get nodes  # Understand cluster state

# Experiment with training parameters
EPOCHS=3 ./run-complete-workflow.sh training
EPOCHS=10 ./run-complete-workflow.sh training  # Compare results

# Test different models
./run-complete-workflow.sh inference  # Test latest model
python scripts/test_mnist_model.py --model output/older_model.pth  # Compare models
```

## 🧪 Testing Different Scenarios

### Test with Your Own Images
```bash
# Single handwritten digit
TEST_IMAGE=my_handwritten_digit.jpg ./run-complete-workflow.sh inference

# Directory of your images
mkdir my_test_digits
# Add your images to my_test_digits/
TEST_IMAGES_DIR=my_test_digits/ ./run-complete-workflow.sh inference
```

### Compare Different Models
```bash
# Train multiple models with different parameters
EPOCHS=5 ./run-complete-workflow.sh training   # Quick model
EPOCHS=20 ./run-complete-workflow.sh training  # Better model

# Test the same images with different models
TEST_IMAGE=my_digit.jpg MODEL_PATH=output/pytorch-single-worker-distributed_TIMESTAMP1/trained-model.pth ./run-complete-workflow.sh inference
TEST_IMAGE=my_digit.jpg MODEL_PATH=output/pytorch-single-worker-distributed_TIMESTAMP2/trained-model.pth ./run-complete-workflow.sh inference
```



### Performance Benchmarking
```bash
# Test inference speed with different image sets
time TEST_IMAGES_DIR=small_set/ ./run-complete-workflow.sh inference
time TEST_IMAGES_DIR=large_set/ ./run-complete-workflow.sh inference
```

## 🐛 Troubleshooting

### Training Issues
```bash
# Check cluster status
kubectl get nodes

# Check training operator
kubectl get pods -n kubeflow

# Check training job
kubectl get pytorchjob
kubectl describe pytorchjob pytorch-single-worker-distributed
```

### Inference Issues
```bash
# Check if model exists
ls -la output/latest/trained-model.pth

# Test dependencies
python -c "import torch, torchvision, PIL, numpy"

# Verify image format
python -c "from PIL import Image; img = Image.open('test_images/digit3.jpg'); print(f'Format: {img.format}, Size: {img.size}, Mode: {img.mode}')"
```

### Common Solutions
- **OOM Errors**: Reduce batch size or worker count
- **Slow Training**: Check CPU/memory limits in YAML configs
- **Image Errors**: Ensure images are grayscale or convert automatically
- **Model Not Found**: Run training first or check output directory

## 🔗 Next Steps

After completing this example:

1. **[Custom Dataset](../03-custom-dataset/)** - Train with your own data
2. **[GPU Training](../04-gpu-training/)** - Accelerate with GPUs  
3. **[Debugging](../05-debugging/)** - Learn debugging techniques
4. **[Existing Clusters](../02-existing-cluster/)** - Use EKS, GKE, AKS

## 📚 Integration Ideas

- **Web App**: Flask/FastAPI with image upload
- **Desktop App**: tkinter with drag-and-drop
- **REST API**: Containerized inference service
- **Mobile**: PyTorch Mobile for on-device inference
- **CI/CD**: Automated model testing pipeline

---

**🎯 This example demonstrates the complete ML workflow from training to deployment!** 