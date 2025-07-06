# 🐛 Example 16: Common Issues & Solutions

**Goal:** Troubleshoot the most frequent problems users encounter with distributed PyTorch training setup.

**What you'll learn:**
- Diagnostic techniques for common failures
- Step-by-step solutions for setup issues
- Prevention strategies for future problems
- When to seek additional help

---

## 🔍 Quick Diagnostics

Run these commands first to gather basic information:

```bash
# Check cluster connectivity
kubectl cluster-info

# Check system resources
kubectl get nodes -o wide
kubectl top nodes

# Check training operator
kubectl get deployment training-operator -n kubeflow

# Check recent events
kubectl get events --sort-by='.lastTimestamp' | tail -20
```

---

## 🚨 Setup Issues

### Issue 1: "kind: command not found"

**Problem:** Kind is not installed or not in PATH.

**Solution:**
```bash
# Install kind manually
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# Verify installation
kind --version
```

**Prevention:** Use `./setup.sh install-deps` for automated installation.

### Issue 2: "Docker daemon not running"

**Problem:** Docker service is not started.

**Solution:**
```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify Docker is running
docker ps
```

**Prevention:** Setup script handles this automatically on most systems.

### Issue 3: "Permission denied" errors

**Problem:** Insufficient permissions for cluster operations.

**Solution:**
```bash
# Check current permissions
kubectl auth can-i create deployments
kubectl auth can-i create customresourcedefinitions

# For existing clusters, get proper credentials
aws eks update-kubeconfig --name your-cluster  # EKS
gcloud container clusters get-credentials your-cluster  # GKE
az aks get-credentials --resource-group rg --name cluster  # AKS
```

**Prevention:** Ensure cluster admin access before setup.

---

## 🏗️ Cluster Issues

### Issue 4: "Cluster creation failed"

**Problem:** Kind cluster creation times out or fails.

**Solution:**
```bash
# Check available resources
free -h
df -h

# Clean up previous clusters
kind delete cluster --name kubeflow-trainer-single-worker
docker system prune -f

# Retry with more time
kind create cluster --config=configs/kind-cluster-config.yaml --wait=600s
```

**Prevention:** Ensure 8GB+ RAM and 10GB+ disk space.

### Issue 5: "Node not ready"

**Problem:** Cluster nodes stuck in NotReady state.

**Solution:**
```bash
# Check node status
kubectl get nodes -o wide
kubectl describe nodes

# Check node conditions
kubectl get nodes -o json | jq '.items[].status.conditions'

# Restart cluster if needed
make delete-cluster
make create-cluster
```

**Prevention:** Ensure stable network and sufficient resources.

---

## 🤖 Training Operator Issues

### Issue 6: "Training operator not found"

**Problem:** Kubeflow training operator not installed or failed.

**Solution:**
```bash
# Check operator deployment
kubectl get deployment training-operator -n kubeflow

# Check operator logs
kubectl logs -n kubeflow -l app=training-operator

# Reinstall operator
./setup.sh install-operator
```

**Prevention:** Always run operator installation step.

### Issue 7: "PyTorchJob CRD not found"

**Problem:** Custom Resource Definition not created.

**Solution:**
```bash
# Check CRD existence
kubectl get crd pytorchjobs.kubeflow.org

# Manually create CRD if needed
kubectl apply -f https://raw.githubusercontent.com/kubeflow/training-operator/master/manifests/base/crds/kubeflow.org_pytorchjobs.yaml

# Verify CRD
kubectl get crd | grep pytorch
```

**Prevention:** Use `./setup.sh install-operator` for proper installation.

---

## 📋 Job Issues

### Issue 8: "Job stuck in Pending"

**Problem:** Training job pods cannot be scheduled.

**Solution:**
```bash
# Check pod status
kubectl get pods -l job-name=pytorch-single-worker-distributed

# Check events
kubectl get events --field-selector involvedObject.name=pytorch-single-worker-distributed

# Check resource availability
kubectl describe nodes
kubectl top nodes

# Reduce resource requests if needed
kubectl edit pytorchjob pytorch-single-worker-distributed
```

**Prevention:** Ensure cluster has sufficient CPU/memory.

### Issue 9: "Job failed with error"

**Problem:** Training job fails during execution.

**Solution:**
```bash
# Check job status
kubectl describe pytorchjob pytorch-single-worker-distributed

# Check pod logs
kubectl logs -l job-name=pytorch-single-worker-distributed --all-containers=true

# Check specific pod
kubectl get pods -l job-name=pytorch-single-worker-distributed
kubectl logs pod-name -c pytorch

# Common fixes:
# 1. Check dataset availability
ls -la input/
# 2. Verify ConfigMap
kubectl get configmap pytorch-training-script -o yaml
# 3. Check image pull
kubectl describe pod pod-name
```

**Prevention:** Validate setup with `./setup.sh validate`.

---

## 🔄 Performance Issues

### Issue 10: "Training very slow"

**Problem:** Training performance is poor.

**Solution:**
```bash
# Check resource utilization
kubectl top pods

# Check if CPU/memory limited
kubectl describe pytorchjob pytorch-single-worker-distributed

# Check network performance
kubectl exec -it pod-name -- ping other-pod-ip

# Optimize resources
kubectl patch pytorchjob pytorch-single-worker-distributed --patch '
spec:
  pytorchReplicaSpecs:
    Master:
      template:
        spec:
          containers:
          - name: pytorch
            resources:
              requests:
                cpu: "1"
                memory: "2Gi"
              limits:
                cpu: "2"
                memory: "4Gi"'
```

**Prevention:** Monitor resources and scale appropriately.

### Issue 11: "Out of memory errors"

**Problem:** Training pods killed due to memory limits.

**Solution:**
```bash
# Check memory usage
kubectl top pods

# Check pod events
kubectl get events --field-selector involvedObject.name=pod-name

# Increase memory limits
kubectl patch pytorchjob pytorch-single-worker-distributed --patch '
spec:
  pytorchReplicaSpecs:
    Master:
      template:
        spec:
          containers:
          - name: pytorch
            resources:
              requests:
                memory: "4Gi"
              limits:
                memory: "8Gi"'
```

**Prevention:** Start with higher memory limits for complex models.

---

## 🌐 Network Issues

### Issue 12: "Connection refused"

**Problem:** Pods cannot communicate with each other.

**Solution:**
```bash
# Check network policies
kubectl get networkpolicies

# Check service endpoints
kubectl get endpoints

# Test connectivity
kubectl exec -it master-pod -- ping worker-pod-ip
kubectl exec -it master-pod -- nc -zv worker-pod-ip 23456

# Check firewall (Linux)
sudo firewall-cmd --list-all
sudo ufw status
```

**Prevention:** Use standard Kubernetes networking.

---

## 🔐 Security Issues

### Issue 13: "RBAC errors"

**Problem:** Role-based access control prevents operations.

**Solution:**
```bash
# Check current permissions
kubectl auth can-i '*' '*'

# Check service account
kubectl get serviceaccounts
kubectl describe serviceaccount default

# For existing clusters, ensure proper roles
kubectl create clusterrolebinding admin-binding \
  --clusterrole=cluster-admin \
  --user=$(kubectl config current-context)
```

**Prevention:** Ensure cluster admin access before setup.

---

## 🔧 Advanced Diagnostics

### Complete System Check

```bash
# Run comprehensive diagnostics
make debug

# Check all cluster components
kubectl get all --all-namespaces

# Check system logs
journalctl -u docker
journalctl -u kubelet

# Check resource quotas
kubectl get resourcequotas --all-namespaces
```

### Recovery Procedures

```bash
# Complete reset (Kind clusters only)
make cleanup-all
make setup
make setup-training

# Partial reset (existing clusters)
make cleanup
kubectl delete namespace kubeflow
./setup.sh install-operator
./setup.sh prepare-training
```

---

## 🆘 When to Seek Help

Seek additional help when:

1. **System-specific issues** not covered here
2. **Cloud provider errors** (AWS, GCP, Azure specific)
3. **Hardware compatibility** problems
4. **Network configuration** issues in enterprise environments
5. **Custom modifications** causing unexpected behavior

### Where to Get Help

- **GitHub Issues** - Report bugs and feature requests
- **Kubernetes Community** - General Kubernetes questions
- **PyTorch Forums** - Training-specific questions
- **Cloud Provider Support** - For managed cluster issues

---

## 📚 Related Examples

- **[01-complete-workflow](../01-complete-workflow/)** - Start fresh if issues persist
- **[02-existing-cluster](../02-existing-cluster/)** - Alternative cluster setup
- **[05-debugging](../05-debugging/)** - Advanced debugging techniques

**🔧 Remember:** Most issues are resolved by ensuring proper setup steps and sufficient resources! 