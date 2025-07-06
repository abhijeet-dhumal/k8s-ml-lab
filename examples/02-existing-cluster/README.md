# 🏗️ Example 02: Existing Cluster

**Goal:** Use your existing Kubernetes cluster (EKS, GKE, AKS, minikube, etc.) for distributed training.

**Prerequisites:**
- Existing Kubernetes cluster
- `kubectl` configured and cluster accessible
- Cluster admin permissions (for operator installation)

**What you'll learn:**
- Cluster detection and validation
- Training operator installation on existing clusters
- Resource compatibility checks
- Safe deployment practices

---

## Step 1: Verify Cluster Access

```bash
# Check cluster connectivity
kubectl cluster-info

# Expected output:
# Kubernetes control plane is running at https://your-cluster-endpoint
# CoreDNS is running at https://your-cluster-endpoint/api/v1/namespaces/kube-system/services/kube-dns:dns/proxy

# Check your permissions
kubectl auth can-i create deployments
kubectl auth can-i create namespaces
kubectl auth can-i create customresourcedefinitions
```

## Step 2: Clone Repository

```bash
git clone https://github.com/<your-username>/distributed-pytorch-training-setup.git
cd distributed-pytorch-training-setup
```

## Step 3: Detect and Validate Cluster

```bash
# Detect cluster type and show information
make debug

# Expected output:
# ✅ Cluster detected: EKS (or GKE, AKS, minikube, etc.)
# ✅ Cluster context: arn:aws:eks:us-west-2:123456789012:cluster/my-cluster
# ✅ Nodes: 3 nodes ready
# ✅ Resources: 12 cores, 48GB RAM available
```

```bash
# Check system requirements
make verify-system

# Expected output:
# ✅ System requirements check passed
# ✅ kubectl is accessible
# ✅ Cluster has sufficient resources
# ✅ Cluster permissions validated
```

## Step 4: Use Existing Cluster

```bash
# Use existing cluster (skips cluster creation)
make use-existing

# Expected output:
# ✅ Using existing cluster: EKS (my-cluster)
# ✅ Cluster validation passed
# ✅ Existing cluster setup completed successfully!
# 
# Next steps:
# 1. Run 'make install-operator' to install training operator
# 2. Run 'make submit-job' to start training
```

## Step 5: Install Training Operator

```bash
# Install Kubeflow training operator
make install-operator

# Expected output:
# ✅ Installing Kubeflow training operator...
# ✅ Training operator deployed to namespace: kubeflow
# ✅ PyTorchJob CRD created successfully
# ✅ Kubeflow training operator installed successfully!
```

**What this does:**
- Creates `kubeflow` namespace
- Deploys training operator
- Creates PyTorchJob custom resource definition
- Validates operator is running

## Step 6: Prepare Training Environment

```bash
# Prepare training environment
make install-operator

# Expected output:
# ✅ MNIST dataset downloaded
# ✅ Training script ConfigMap created
# ✅ Training environment prepared successfully!
```

## Step 7: Submit Training Job

```bash
# Submit distributed training job
make submit-job

# Expected output:
# ✅ Job submitted: pytorch-single-worker-distributed
# Use 'make status' to check job status
```

## Step 8: Monitor Training

```bash
# Check job status
make status

# View training logs
make logs

# Monitor in real-time
make watch-job
```

---

## 🎯 Cluster-Specific Examples

### AWS EKS

```bash
# Connect to EKS cluster
aws eks update-kubeconfig --region us-west-2 --name my-cluster

# Use existing cluster
make use-existing

# Expected cluster info:
# Cluster Type: EKS
# Region: us-west-2
# Node Groups: 2-3 groups
# Instance Types: m5.large, m5.xlarge, etc.
```

### Google GKE

```bash
# Connect to GKE cluster
gcloud container clusters get-credentials my-cluster --zone us-central1-a

# Use existing cluster
make use-existing

# Expected cluster info:
# Cluster Type: GKE
# Zone: us-central1-a
# Node Pools: 1-2 pools
# Machine Types: n1-standard-2, n1-standard-4, etc.
```

### Azure AKS

```bash
# Connect to AKS cluster
az aks get-credentials --resource-group myResourceGroup --name myCluster

# Use existing cluster
make use-existing

# Expected cluster info:
# Cluster Type: AKS
# Resource Group: myResourceGroup
# Node Pools: 1-2 pools
# VM Sizes: Standard_D2s_v3, Standard_D4s_v3, etc.
```

### Minikube

```bash
# Start minikube (if not running)
minikube start --memory=8192 --cpus=4

# Use existing cluster
make use-existing

# Expected cluster info:
# Cluster Type: minikube
# Driver: docker (or virtualbox)
# Resources: 4 CPUs, 8GB RAM
```

## 🔍 Resource Validation

The setup script validates your cluster has sufficient resources:

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **CPU** | 2 cores | 4+ cores |
| **Memory** | 4GB | 8GB+ |
| **Nodes** | 1 | 2+ |
| **Storage** | 10GB | 20GB+ |

```bash
# Check resource availability
kubectl top nodes
kubectl describe nodes
```

## ⚠️ Safety Features

### Cluster Protection
- **Never deletes existing clusters** - Only creates/manages training jobs
- **Namespace isolation** - Uses `kubeflow` namespace for operator
- **Resource limits** - Training jobs have CPU/memory limits
- **RBAC respect** - Uses existing cluster permissions

### Safe Commands for Existing Clusters
```bash
# ✅ Safe to use with existing clusters
make submit-job
make status
make logs
make cleanup          # Only removes training resources

# ⚠️ NEVER use these with existing clusters
make delete-cluster   # Will try to delete your cluster!
make cleanup-all      # Includes cluster deletion
make reset           # Recreates cluster
```

## 🐛 Common Issues

### Issue: "Insufficient permissions"
```bash
# Check required permissions
kubectl auth can-i create deployments --namespace kubeflow
kubectl auth can-i create customresourcedefinitions

# Solution: Get cluster admin access or ask administrator
```

### Issue: "Resources not available"
```bash
# Check node resources
kubectl top nodes
kubectl describe nodes

# Solution: Scale up cluster or reduce job resource requests
```

### Issue: "Training operator installation failed"
```bash
# Check operator logs
kubectl logs -n kubeflow -l app=training-operator

# Check CRD creation
kubectl get crd pytorchjobs.kubeflow.org

# Solution: Ensure cluster admin permissions
```

### Issue: "Pod scheduling failed"
```bash
# Check pod events
kubectl get events --sort-by='.lastTimestamp' | tail -10

# Check resource requests
kubectl describe pytorchjob pytorch-single-worker-distributed

# Solution: Adjust resource requests or scale cluster
```

## 🧹 Cleanup

```bash
# Clean up training resources only (safe for existing clusters)
make cleanup

# Remove training operator (optional)
kubectl delete namespace kubeflow
```

## 📊 Cluster Comparison

| Cluster Type | Setup Time | Cost | Scalability | Best For |
|-------------|------------|------|-------------|----------|
| **EKS** | 15-20 min | $$$ | High | Production |
| **GKE** | 10-15 min | $$$ | High | Production |
| **AKS** | 15-20 min | $$$ | High | Production |
| **Minikube** | 5 min | Free | Low | Development |
| **Kind** | 2 min | Free | Low | Testing |

## 📚 Next Steps

After completing this example:

1. **[04-gpu-training](../04-gpu-training/)** - Add GPU nodes for acceleration
2. **[03-custom-dataset](../03-custom-dataset/)** - Train with your own data
3. **[05-debugging](../05-debugging/)** - Learn debugging techniques
4. **[06-common-issues](../06-common-issues/)** - Troubleshooting guide

---

**🎉 Success!** You've successfully integrated distributed PyTorch training with your existing Kubernetes cluster. Your production environment is now ready for ML workloads! 