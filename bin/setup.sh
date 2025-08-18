#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
CLUSTER_NAME="pytorch-training-cluster"
KUBEFLOW_VERSION="v1.7.0"
KUBECTL_VERSION="v1.28.0"
KIND_VERSION="v0.20.0"

# Workflow configuration
EPOCHS=${EPOCHS:-5}
WORKERS=${WORKERS:-2}
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.001}
JOB_TIMEOUT=${JOB_TIMEOUT:-600}  # 10 minutes
JOB_NAME="mnist-training"

# Cluster type detection
USE_EXISTING_CLUSTER=false
CLUSTER_TYPE=""
CLUSTER_CONTEXT=""

log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}"
    exit 1
}

# Check if running on macOS or Linux
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="darwin"
        ARCH="amd64"
        if [[ $(uname -m) == "arm64" ]]; then
            ARCH="arm64"
        fi
        DISTRO="macOS"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        ARCH="amd64"
        if [[ $(uname -m) == "aarch64" ]]; then
            ARCH="arm64"
        fi
        
        # Detect Linux distribution
        if [[ -f /etc/os-release ]]; then
            . /etc/os-release
            DISTRO="$NAME"
            DISTRO_ID="$ID"
            DISTRO_VERSION="$VERSION_ID"
        elif [[ -f /etc/redhat-release ]]; then
            DISTRO=$(cat /etc/redhat-release)
            DISTRO_ID="rhel"
        elif [[ -f /etc/lsb-release ]]; then
            . /etc/lsb-release
            DISTRO="$DISTRIB_ID"
            DISTRO_ID="$DISTRIB_ID"
        else
            DISTRO="Unknown Linux"
            DISTRO_ID="unknown"
        fi
    else
        error "Unsupported OS: $OSTYPE"
    fi
    log "Detected OS: $OS/$ARCH"
    log "Distribution: $DISTRO"
}

# Detect package manager
detect_package_manager() {
    if command -v dnf &> /dev/null; then
        PKG_MANAGER="dnf"
        PKG_INSTALL="dnf install -y"
        PKG_UPDATE="dnf update -y"
    elif command -v yum &> /dev/null; then
        PKG_MANAGER="yum"
        PKG_INSTALL="yum install -y"
        PKG_UPDATE="yum update -y"
    elif command -v apt-get &> /dev/null; then
        PKG_MANAGER="apt"
        PKG_INSTALL="apt-get install -y"
        PKG_UPDATE="apt-get update"
    elif command -v pacman &> /dev/null; then
        PKG_MANAGER="pacman"
        PKG_INSTALL="pacman -S --noconfirm"
        PKG_UPDATE="pacman -Sy"
    elif command -v brew &> /dev/null; then
        PKG_MANAGER="brew"
        PKG_INSTALL="brew install"
        PKG_UPDATE="brew update"
    else
        error "No supported package manager found (dnf, yum, apt-get, pacman, brew)"
    fi
    log "Package manager: $PKG_MANAGER"
}

# Check system requirements
check_system_requirements() {
    log "Checking system requirements..."
    
    # Check memory (minimum 8GB, recommended 16GB)
    local mem_gb=0
    if [[ "$OS" == "linux" ]]; then
        local mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
        mem_gb=$((mem_kb / 1024 / 1024))
    elif [[ "$OS" == "darwin" ]]; then
        local mem_bytes=$(sysctl -n hw.memsize)
        mem_gb=$((mem_bytes / 1024 / 1024 / 1024))
    fi
    
    if [[ $mem_gb -gt 0 ]]; then
        if [[ $mem_gb -lt 8 ]]; then
            error "Insufficient memory: ${mem_gb}GB found, minimum 8GB required"
        elif [[ $mem_gb -lt 16 ]]; then
            warn "Memory: ${mem_gb}GB (16GB recommended for optimal performance)"
        else
            success "Memory: ${mem_gb}GB (sufficient)"
        fi
    else
        warn "Memory: Unable to detect memory size"
    fi
    
    # Check CPU cores (minimum 4 cores)
    local cpu_cores=0
    if [[ "$OS" == "linux" ]]; then
        cpu_cores=$(nproc)
    elif [[ "$OS" == "darwin" ]]; then
        cpu_cores=$(sysctl -n hw.ncpu)
    fi
    
    if [[ $cpu_cores -gt 0 ]]; then
        if [[ $cpu_cores -lt 4 ]]; then
            error "Insufficient CPU cores: ${cpu_cores} found, minimum 4 required"
        else
            success "CPU cores: ${cpu_cores} (sufficient)"
        fi
    else
        warn "CPU cores: Unable to detect CPU cores"
    fi
    
    # Check disk space (minimum 10GB free)
    local disk_space_gb=0
    if [[ "$OS" == "linux" ]]; then
        disk_space_gb=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    elif [[ "$OS" == "darwin" ]]; then
        # macOS df output format is different, use -h and parse
        local disk_space_raw=$(df -h . | awk 'NR==2 {print $4}')
        if [[ $disk_space_raw == *"G"* ]]; then
            disk_space_gb=$(echo $disk_space_raw | sed 's/G.*//')
        elif [[ $disk_space_raw == *"T"* ]]; then
            local disk_space_tb=$(echo $disk_space_raw | sed 's/T.*//')
            disk_space_gb=$((disk_space_tb * 1024))
        fi
    fi
    
    if [[ $disk_space_gb -gt 0 ]]; then
        if [[ $disk_space_gb -lt 10 ]]; then
            error "Insufficient disk space: ${disk_space_gb}GB free, minimum 10GB required"
        else
            success "Disk space: ${disk_space_gb}GB free (sufficient)"
        fi
    else
        warn "Disk space: Unable to detect free disk space"
    fi
    
    success "System requirements check passed"
}

# Install container runtime (Docker or Podman)
install_container_runtime() {
    if command -v docker &> /dev/null; then
        success "Docker already installed"
        # Check if Docker daemon is running
        if ! docker info &> /dev/null; then
            log "Starting Docker daemon..."
            sudo systemctl start docker 2>/dev/null || true
            sudo systemctl enable docker 2>/dev/null || true
        fi
        return
    fi
    
    if command -v podman &> /dev/null; then
        success "Podman already installed"
        return
    fi
    
    log "Installing container runtime..."
    
    case "$DISTRO_ID" in
        "fedora")
            log "Installing Docker on Fedora..."
            sudo $PKG_INSTALL dnf-plugins-core
            sudo dnf config-manager --add-repo https://download.docker.com/linux/fedora/docker-ce.repo
            sudo $PKG_INSTALL docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        "rhel"|"centos")
            log "Installing Docker on RHEL/CentOS..."
            sudo $PKG_INSTALL yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo $PKG_INSTALL docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        "ubuntu"|"debian")
            log "Installing Docker on Ubuntu/Debian..."
            sudo $PKG_UPDATE
            sudo $PKG_INSTALL ca-certificates curl gnupg lsb-release
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo $PKG_UPDATE
            sudo $PKG_INSTALL docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        "arch")
            log "Installing Docker on Arch Linux..."
            sudo $PKG_INSTALL docker
            sudo systemctl start docker
            sudo systemctl enable docker
            sudo usermod -aG docker $USER
            ;;
        *)
            warn "Unknown distribution, trying to install Docker via package manager..."
            sudo $PKG_INSTALL docker || sudo $PKG_INSTALL podman
            ;;
    esac
    
    success "Container runtime installed"
    warn "Please log out and log back in for Docker group membership to take effect"
}

# Install Python and pip
install_python() {
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 --version | awk '{print $2}')
        success "Python3 already installed: $py_version"
    else
        log "Installing Python3..."
        case "$DISTRO_ID" in
            "fedora"|"rhel"|"centos")
                sudo $PKG_INSTALL python3 python3-pip python3-devel
                ;;
            "ubuntu"|"debian")
                sudo $PKG_UPDATE
                sudo $PKG_INSTALL python3 python3-pip python3-dev
                ;;
            "arch")
                sudo $PKG_INSTALL python python-pip
                ;;
            *)
                sudo $PKG_INSTALL python3 python3-pip
                ;;
        esac
        success "Python3 installed"
    fi
    
    if command -v pip3 &> /dev/null; then
        success "pip3 already installed"
    else
        log "Installing pip3..."
        case "$DISTRO_ID" in
            "fedora"|"rhel"|"centos")
                sudo $PKG_INSTALL python3-pip
                ;;
            "ubuntu"|"debian")
                sudo $PKG_INSTALL python3-pip
                ;;
            "arch")
                sudo $PKG_INSTALL python-pip
                ;;
            *)
                sudo $PKG_INSTALL python3-pip
                ;;
        esac
        success "pip3 installed"
    fi
}

# Install kubectl
install_kubectl() {
    if command -v kubectl &> /dev/null; then
        local current_version=$(kubectl version --client --short 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | head -1)
        success "kubectl already installed: $current_version"
        return
    fi
    
    log "Installing kubectl $KUBECTL_VERSION..."
    curl -LO "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/$OS/$ARCH/kubectl"
    chmod +x kubectl
    sudo mv kubectl /usr/local/bin/
    success "kubectl installed successfully"
}

# Install kind
install_kind() {
    if command -v kind &> /dev/null; then
        local current_version=$(kind version | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+')
        success "kind already installed: $current_version"
        return
    fi
    
    log "Installing kind $KIND_VERSION..."
    curl -Lo ./kind "https://kind.sigs.k8s.io/dl/$KIND_VERSION/kind-$OS-$ARCH"
    chmod +x ./kind
    sudo mv ./kind /usr/local/bin/kind
    success "kind installed successfully"
}

# Install Python dependencies
install_python_deps() {
    log "Installing Python dependencies..."
    
    # Check if pip is available
    if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
        error "pip not found. Please install Python 3 and pip first."
    fi
    
    # Use pip3 if available, otherwise pip
    local pip_cmd="pip3"
    if ! command -v pip3 &> /dev/null; then
        pip_cmd="pip"
    fi
    
    # Install PyTorch and related packages
    $pip_cmd install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    $pip_cmd install kubernetes pyyaml
    
    success "Python dependencies installed"
}

# Detect existing Kubernetes cluster
detect_existing_cluster() {
    log "Detecting existing Kubernetes cluster..."
    
    if ! command -v kubectl &> /dev/null; then
        log "kubectl not found - will install"
        return
    fi
    
    # Check if kubectl can connect to a cluster
    if kubectl cluster-info &> /dev/null; then
        CLUSTER_CONTEXT=$(kubectl config current-context 2>/dev/null)
        local cluster_info=$(kubectl cluster-info 2>/dev/null)
        
        # Check context name first, then cluster info
        if [[ $CLUSTER_CONTEXT == *"kind"* ]] || [[ $cluster_info == *"kind"* ]]; then
            CLUSTER_TYPE="kind"
        elif [[ $CLUSTER_CONTEXT == *"minikube"* ]] || [[ $cluster_info == *"minikube"* ]]; then
            CLUSTER_TYPE="minikube"
        elif [[ $CLUSTER_CONTEXT == *"k3s"* ]] || [[ $cluster_info == *"k3s"* ]]; then
            CLUSTER_TYPE="k3s"
        elif [[ $cluster_info == *"eks"* ]] || [[ $CLUSTER_CONTEXT == *"eks"* ]]; then
            CLUSTER_TYPE="EKS"
        elif [[ $cluster_info == *"gke"* ]] || [[ $CLUSTER_CONTEXT == *"gke"* ]]; then
            CLUSTER_TYPE="GKE"
        elif [[ $cluster_info == *"aks"* ]] || [[ $CLUSTER_CONTEXT == *"aks"* ]]; then
            CLUSTER_TYPE="AKS"
        else
            CLUSTER_TYPE="unknown"
        fi
        
        success "Found existing cluster: $CLUSTER_TYPE"
        success "Current context: $CLUSTER_CONTEXT"
        
        # Check cluster compatibility
        check_cluster_compatibility
        
        return 0
    else
        log "No accessible cluster found - will create new kind cluster"
        return 1
    fi
}

# Check cluster compatibility
check_cluster_compatibility() {
    log "Checking cluster compatibility..."
    
    # Check if cluster has at least 1 node
    local node_count=$(kubectl get nodes --no-headers 2>/dev/null | wc -l)
    if [[ $node_count -lt 1 ]]; then
        error "No nodes found in cluster"
    fi
    
    # Check if cluster has sufficient resources
    local total_cpu=0
    local total_memory=0
    
    # Get allocatable resources from all nodes
    while IFS= read -r line; do
        local cpu=$(echo "$line" | awk '{print $1}' | sed 's/m$//')
        local memory=$(echo "$line" | awk '{print $2}' | sed 's/Ki$//')
        total_cpu=$((total_cpu + cpu))
        total_memory=$((total_memory + memory))
    done < <(kubectl get nodes -o custom-columns=CPU:.status.allocatable.cpu,MEMORY:.status.allocatable.memory --no-headers 2>/dev/null)
    
    local total_memory_gb=$((total_memory / 1024 / 1024))
    local total_cpu_cores=$((total_cpu / 1000))
    
    success "Cluster resources: ${total_cpu_cores} CPU cores, ${total_memory_gb}GB memory"
    
    # Check minimum requirements
    if [[ $total_cpu_cores -lt 1 ]]; then
        error "Insufficient CPU in cluster: ${total_cpu_cores} cores available, minimum 1 required"
    fi
    
    if [[ $total_memory_gb -lt 2 ]]; then
        error "Insufficient memory in cluster: ${total_memory_gb}GB available, minimum 2GB required"
    fi
    
    # Check if cluster supports creating namespaces
    if ! kubectl auth can-i create namespaces &> /dev/null; then
        warn "Cannot create namespaces - may need cluster admin privileges"
    fi
    
    success "Cluster compatibility check passed"
}

# Prompt user for cluster choice
prompt_cluster_choice() {
    if detect_existing_cluster; then
        echo ""
        echo "🎯 Cluster Detection Results:"
        echo "=============================="
        echo "Found existing cluster: $CLUSTER_TYPE"
        echo "Current context: $CLUSTER_CONTEXT"
        echo ""
        echo "Options:"
        echo "1. Use existing cluster (recommended if compatible)"
        echo "2. Create new kind cluster"
        echo "3. Exit and configure cluster manually"
        echo ""
        
        read -p "Choose option (1-3): " choice
        
        case $choice in
            1)
                USE_EXISTING_CLUSTER=true
                success "Using existing cluster: $CLUSTER_TYPE"
                ;;
            2)
                USE_EXISTING_CLUSTER=false
                success "Will create new kind cluster"
                ;;
            3)
                log "Exiting. You can run specific setup commands later:"
                echo "  make use-existing           # Use existing cluster"
                echo "  make verify-system          # Check system dependencies"
                exit 0
                ;;
            *)
                error "Invalid choice. Please run the script again."
                ;;
        esac
    else
        USE_EXISTING_CLUSTER=false
        log "No existing cluster found - will create new kind cluster"
    fi
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p input output
    success "Directories created: input/, output/"
}

# Create kind cluster
create_cluster() {
    if [[ "$USE_EXISTING_CLUSTER" == "true" ]]; then
        log "Using existing cluster: $CLUSTER_TYPE ($CLUSTER_CONTEXT)"
        success "Existing cluster will be used"
        return
    fi
    
    if kind get clusters | grep -q "^$CLUSTER_NAME$"; then
        warn "Cluster $CLUSTER_NAME already exists"
        return
    fi
    
    # Ensure directories exist before creating cluster (required for volume mounts)
    log "Ensuring directories exist for Kind volume mounts..."
    mkdir -p input output scripts
    
    log "Creating kind cluster: $CLUSTER_NAME"
    kind create cluster --config=configs/kind-cluster-config.yaml --wait=300s
    
    # Wait for cluster to be ready
    log "Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=300s
    
    success "Kind cluster created and ready"
}

# Install Kubeflow Training Operator
install_kubeflow_operator() {
    log "Installing Kubeflow Training Operator..."
    
    # Install cert-manager first (required for training operator)
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
    
    # Wait for cert-manager to be ready
    log "Waiting for cert-manager to be ready..."
    kubectl wait --for=condition=Available --timeout=300s deployment/cert-manager -n cert-manager
    kubectl wait --for=condition=Available --timeout=300s deployment/cert-manager-cainjector -n cert-manager
    kubectl wait --for=condition=Available --timeout=300s deployment/cert-manager-webhook -n cert-manager
    
    # Install training operator using kustomize
    kubectl apply -k https://github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=$KUBEFLOW_VERSION
    
    # Wait for training operator to be ready
    log "Waiting for training operator to be ready..."
    kubectl wait --for=condition=Available --timeout=300s deployment/training-operator -n kubeflow
    
    success "Kubeflow Training Operator installed and ready"
}

# Create ConfigMap for training script
create_configmap() {
    log "Creating ConfigMap for training script..."
    kubectl create configmap pytorch-training-script \
        --from-file=mnist.py=scripts/mnist.py \
        --dry-run=client -o yaml | kubectl apply -f -
    success "ConfigMap created"
}

# Download MNIST dataset
download_mnist() {
    log "Downloading MNIST dataset..."
    
    # Create a simple Python script to download MNIST
    cat > download_mnist.py << 'EOF'
import os
import torch
from torchvision import datasets, transforms

# Ensure input directory exists
os.makedirs('input', exist_ok=True)

# Download MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

print("Downloading MNIST training dataset...")
train_dataset = datasets.MNIST('input', train=True, download=True, transform=transform)
print(f"Training dataset size: {len(train_dataset)}")

print("Downloading MNIST test dataset...")
test_dataset = datasets.MNIST('input', train=False, download=True, transform=transform)
print(f"Test dataset size: {len(test_dataset)}")

print("MNIST dataset downloaded successfully!")
EOF
    
    python3 download_mnist.py
    rm download_mnist.py
    
    success "MNIST dataset downloaded to input/"
}

# Validate cluster setup
validate_setup() {
    log "Validating cluster setup..."
    
    # Check cluster status
    if ! kubectl cluster-info &> /dev/null; then
        error "Cluster is not accessible"
    fi
    
    # Check nodes
    local node_count=$(kubectl get nodes --no-headers | wc -l)
    if [ "$node_count" -lt 2 ]; then
        error "Expected at least 2 nodes, found $node_count"
    fi
    
    # Check training operator
    if ! kubectl get deployment training-operator -n kubeflow &> /dev/null; then
        error "Training operator not found"
    fi
    
    # Check ConfigMap
    if ! kubectl get configmap pytorch-training-script &> /dev/null; then
        error "Training script ConfigMap not found"
    fi
    
    # Check PyTorchJob CRD
    if ! kubectl get crd pytorchjobs.kubeflow.org &> /dev/null; then
        error "PyTorchJob CRD not found"
    fi
    
    success "All validations passed!"
}

# Print cluster info
print_cluster_info() {
    log "Cluster Information:"
    echo "==================="
    echo "Cluster: $CLUSTER_NAME"
    echo "Nodes:"
    kubectl get nodes -o wide
    echo ""
    echo "Training Operator Status:"
    kubectl get deployment training-operator -n kubeflow
    echo ""
    echo "Available PyTorchJob CRDs:"
    kubectl get crd | grep pytorchjobs
    echo ""
    echo "Input directory contents:"
    ls -la input/ || true
    echo ""
    success "Setup completed successfully! Use 'make submit-job' to run training."
}

    # Install only dependencies (Docker/Podman, Python, kubectl, kind)
install_deps() {
    log "Installing dependencies only..."
    
    detect_os
    detect_package_manager
    check_system_requirements
    install_container_runtime
    install_python
    install_kubectl
    install_kind
    install_python_deps
    
    success "Dependencies installed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'make setup' to create cluster and complete setup"
    echo "2. Or run individual commands:"
    echo "   - 'make use-existing' to use existing cluster"
    echo "   - 'make submit-job' to start training"
}

# Create only cluster (assumes dependencies are installed)
create_cluster_only() {
    log "Creating cluster only..."
    
    detect_os
    
    # Detect existing cluster and prompt user
    prompt_cluster_choice
    
    create_directories
    create_cluster
    
    success "Cluster setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'make install-operator' to install Kubeflow training operator"
    echo "2. Run 'make submit-job' to start training"
    echo "3. Run 'make status' to check job status"
    echo "4. Run 'make logs' to view training logs"
}

# Use existing cluster (assumes cluster is accessible)
use_existing_cluster() {
    log "Using existing cluster..."
    
    detect_os
    
    # Force use of existing cluster
    if detect_existing_cluster; then
        USE_EXISTING_CLUSTER=true
        success "Using existing cluster: $CLUSTER_TYPE ($CLUSTER_CONTEXT)"
    else
        error "No accessible cluster found. Please ensure kubectl is configured and cluster is accessible."
    fi
    
    create_directories
    create_cluster  # This will skip kind creation
    
    success "Existing cluster setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'make install-operator' to install Kubeflow training operator"
    echo "2. Run 'make submit-job' to start training"
    echo "3. Run 'make status' to check job status"
    echo "4. Run 'make logs' to view training logs"
}

# Install Kubeflow training operator only
install_operator_only() {
    log "Installing Kubeflow training operator..."
    
    detect_os
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        error "No accessible cluster found. Please ensure kubectl is configured and cluster is accessible."
    fi
    
    # Detect cluster type for information
    detect_existing_cluster &> /dev/null || true
    
    install_kubeflow_operator
    
    success "Kubeflow training operator installed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Setup is already complete with the operator - ready for training"
    echo "2. Run 'make submit-job' to start training"
}

# Prepare training environment (ConfigMap and dataset)
prepare_training_env() {
    log "Preparing training environment..."
    
    detect_os
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        error "No accessible cluster found. Please ensure kubectl is configured and cluster is accessible."
    fi
    
    # Check if training operator is installed
    if ! kubectl get crd pytorchjobs.kubeflow.org &> /dev/null; then
        error "PyTorchJob CRD not found. Please install the training operator first: make install-operator"
    fi
    
    # Install Python dependencies if not already installed
    install_python_deps
    
    create_directories
    create_configmap
    download_mnist
    validate_setup
    print_cluster_info
    
    success "Training environment prepared successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'make submit-job' to start training"
    echo "2. Run 'make status' to check job status"
    echo "3. Run 'make logs' to view training logs"
}

# Complete training setup (operator + environment)
setup_training() {
    log "Setting up complete training environment..."
    
    detect_os
    
    # Check if cluster is accessible
    if ! kubectl cluster-info &> /dev/null; then
        error "No accessible cluster found. Please ensure kubectl is configured and cluster is accessible."
    fi
    
    # Detect cluster type for information
    detect_existing_cluster &> /dev/null || true
    
    # Install Python dependencies if not already installed
    install_python_deps
    
    install_kubeflow_operator
    create_directories
    create_configmap
    download_mnist
    validate_setup
    print_cluster_info
    
    success "Training environment setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'make submit-job' to start training"
    echo "2. Run 'make status' to check job status"
    echo "3. Run 'make logs' to view training logs"
    echo "4. Run 'make cleanup' to clean up resources"
}

# Main setup function
main() {
    log "Starting distributed PyTorch training setup..."
    
    detect_os
    detect_package_manager
    check_system_requirements
    install_container_runtime
    install_python
    install_kubectl
    
    # Detect existing cluster and prompt user
    prompt_cluster_choice
    
    # Only install kind if we're not using existing cluster
    if [[ "$USE_EXISTING_CLUSTER" == "false" ]]; then
        install_kind
    fi
    
    install_python_deps
    create_directories
    create_cluster
    
    success "Infrastructure setup completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. Run 'make install-operator' to install operator and complete setup"
    echo "2. Or use existing cluster with:"
    echo "   - 'make use-existing' to configure existing cluster"
    echo "   - 'make install-operator' to install training operator"
    echo "3. Finally run 'make submit-job' to start training"
}

# Check system requirements only
check_requirements() {
    log "Checking system requirements only..."
    
    detect_os
    detect_package_manager
    check_system_requirements
    
    success "System requirements check completed!"
}

# Comprehensive system verification
verify_system() {
    log "Running comprehensive system verification..."
    
    detect_os
    detect_package_manager
    check_system_requirements
    
    # Check all required dependencies
    local all_deps_ok=true
    
    # Check container runtime
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            success "Docker: installed and running"
        else
            warn "Docker: installed but not running"
            all_deps_ok=false
        fi
    elif command -v podman &> /dev/null; then
        success "Podman: installed"
    else
        warn "Container runtime: neither Docker nor Podman found"
        all_deps_ok=false
    fi
    
    # Check Python
    if command -v python3 &> /dev/null; then
        local py_version=$(python3 --version 2>&1 | awk '{print $2}')
        success "Python: $py_version"
    else
        warn "Python: not found"
        all_deps_ok=false
    fi
    
    # Check kubectl
    if command -v kubectl &> /dev/null; then
        local kubectl_version=$(kubectl version --client --short 2>/dev/null | awk '{print $3}' || echo "unknown")
        success "kubectl: $kubectl_version"
    else
        warn "kubectl: not found"
        all_deps_ok=false
    fi
    
    # Check kind
    if command -v kind &> /dev/null; then
        local kind_version=$(kind version 2>/dev/null | awk '{print $2}' || echo "unknown")
        success "kind: $kind_version"
    else
        warn "kind: not found"
        all_deps_ok=false
    fi
    
    # Check Python dependencies
    if python3 -c "import torch, torchvision, requests, yaml" &> /dev/null; then
        success "Python dependencies: PyTorch, torchvision, requests, PyYAML installed"
    else
        warn "Python dependencies: some required packages missing"
        all_deps_ok=false
    fi
    
    # Check cluster accessibility (if configured)
    if kubectl cluster-info &> /dev/null; then
        success "Kubernetes cluster: accessible"
        
        # Check if Kubeflow operator is installed
        if kubectl get crd pytorchjobs.kubeflow.org &> /dev/null; then
            success "Kubeflow Training Operator: installed"
        else
            warn "Kubeflow Training Operator: not installed"
        fi
    else
        warn "Kubernetes cluster: not accessible (this is OK if not set up yet)"
    fi
    
    echo ""
    if [[ "$all_deps_ok" == "true" ]]; then
        success "System verification completed - all dependencies are ready!"
        echo ""
        echo "Next steps:"
        echo "1. Run 'make submit-job' to start training"
        echo "2. Or run 'make run-e2e-workflow' for complete workflow"
    else
        warn "System verification found missing dependencies"
        echo ""
        echo "To install missing dependencies:"
        echo "1. Run 'make install-deps' to install basic dependencies"
        echo "2. Run 'make setup-training' to install operators and prepare environment"
        echo "3. Run 'make verify-system' again to recheck"
    fi
}

# ==============================================================================
# WORKFLOW FUNCTIONS
# ==============================================================================

# Enhanced section function for better formatting
section() {
    echo
    echo -e "${BLUE}🚀 $1${NC}"
    echo -e "${BLUE}$(printf '=%.0s' {1..50})${NC}"
}

# Submit and monitor training job
submit_and_monitor_job() {
    section "Distributed Training Job"

    echo "Training Configuration:"
    echo "  📊 Epochs: $EPOCHS"
    echo "  👥 Workers: $WORKERS" 
    echo "  📦 Batch Size: $BATCH_SIZE"
    echo "  📈 Learning Rate: $LEARNING_RATE"
    echo "  ⏱️  Timeout: $JOB_TIMEOUT seconds"
    echo

    log "Cleaning up any existing jobs..."
    kubectl delete pytorchjob "$JOB_NAME" --ignore-not-found=true
    sleep 5

    log "Preparing required directories..."
    mkdir -p input output scripts
    
    # Ensure directories exist for Kind volume mounts
    if [[ ! -d "scripts" ]]; then
        error "scripts/ directory not found. Required for Kind volume mount."
    fi
    if [[ ! -d "input" ]]; then
        warn "input/ directory not found. Creating it now..."
        mkdir -p input
    fi
    if [[ ! -d "output" ]]; then
        warn "output/ directory not found. Creating it now..."
        mkdir -p output
    fi
    
    # Check if training script exists
        if [[ ! -f "scripts/mnist.py" ]]; then
        error "Training script not found: scripts/mnist.py"
    fi
    
    # Download MNIST dataset if not already present
    if [[ ! -d "input/MNIST" ]]; then
        log "Downloading MNIST dataset..."
        download_mnist
    else
        log "MNIST dataset already exists in input/"
    fi
    
    log "All required directories ready:"
    ls -la scripts/ input/ output/ | head -10

    log "Submitting distributed training job..."
    kubectl apply -f configs/pytorch-distributed-job.yaml

    success "Training job submitted successfully!"
}

# Monitor job progress
monitor_job() {
    section "Job Progress Monitoring"

    log "Monitoring training progress... (detailed status every 30s)"
    counter=0
    while true; do
        # Check if job exists
        if ! kubectl get pytorchjob "$JOB_NAME" &> /dev/null; then
            error "Training job not found"
        fi
        
        # Get job status
        status=$(kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.status.conditions[?(@.type=="Succeeded")].status}' 2>/dev/null || echo "")
        failed_status=$(kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
        
        # Check completion
        if [[ "$status" == "True" ]]; then
            success "PyTorchJob completed successfully!"
            
            # Verify training actually succeeded by checking pod logs
            log "Verifying training success..."
            master_pod=$(kubectl get pods -l training.kubeflow.org/job-name="$JOB_NAME",training.kubeflow.org/replica-type=master -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
            if [[ -n "$master_pod" ]]; then
                # Check if training script completed successfully
                if kubectl logs "$master_pod" | grep -q "Distributed training completed successfully"; then
                    success "Training script completed successfully!"
                else
                    warn "PyTorchJob succeeded but training script may have issues. Check logs."
                fi
                
                # Check if model was saved
                if kubectl exec "$master_pod" -- test -f /output/trained-model.pth 2>/dev/null; then
                    success "Model file found in pod!"
                else
                    warn "Model file not found in pod - training may not have saved properly"
                fi
            else
                warn "Cannot verify training success - master pod not found"
            fi
            
            echo ""  # New line after progress dots
            break
        elif [[ "$failed_status" == "True" ]]; then
            echo ""  # New line after progress dots
            # Get failure reason
            failure_reason=$(kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.status.conditions[?(@.type=="Failed")].message}' 2>/dev/null || echo "Unknown")
            error "Training job failed: $failure_reason"
        elif [[ $counter -ge $JOB_TIMEOUT ]]; then
            echo ""  # New line after progress dots
            error "Training timeout after $JOB_TIMEOUT seconds"
        fi
        
        # Show minimal progress
        if [[ $((counter % 30)) -eq 0 ]]; then
            current_status=$(kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.status.conditions[?(@.type=="Running")].status}' 2>/dev/null || echo "")
            job_state=$(kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.metadata.name}' 2>/dev/null && echo " (Running)" || echo " (Starting...)")
            echo "⏱️  Training: ${counter}s - Status: ${job_state}"
            
            # Only show pod details if there are issues
            failing_pods=$(kubectl get pods -l training.kubeflow.org/job-name="$JOB_NAME" --field-selector=status.phase=Failed -o name 2>/dev/null || true)
            pending_pods=$(kubectl get pods -l training.kubeflow.org/job-name="$JOB_NAME" --field-selector=status.phase=Pending -o name 2>/dev/null || true)
            
            if [[ -n "$failing_pods" ]]; then
                warn "Failed pods detected: $failing_pods"
            elif [[ -n "$pending_pods" ]]; then
                warn "Pods still pending: $pending_pods"
            fi
        else
            # Simple progress dots
            echo -n "."
        fi
        
        sleep 15
        counter=$((counter + 15))
    done
}

# Collect training artifacts
collect_artifacts() {
    section "Collecting Training Artifacts"

    # Create clean, organized directory structure
    timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
    
    # Main organized structure
    models_dir="output/models"
    logs_dir="output/logs"
    archive_dir="output/archive"
    
    # Latest directories
    latest_models_dir="$models_dir/latest"
    latest_logs_dir="$logs_dir/latest"
    
    # Archive directories for this run
    run_archive_dir="$archive_dir/${JOB_NAME}_${timestamp}"
    
    # Create directory structure
    mkdir -p "$latest_models_dir" "$latest_logs_dir" "$run_archive_dir"
    
    # Brief final status
    log "Final training status:"
    kubectl get pytorchjob "$JOB_NAME" -o custom-columns=NAME:.metadata.name,STATE:.status.conditions[-1].type,AGE:.metadata.creationTimestamp --no-headers 2>/dev/null || true

    # Collect logs from pods
    log "Collecting pod logs..."
    kubectl logs -l training.kubeflow.org/job-name="$JOB_NAME",training.kubeflow.org/replica-type=master > "$latest_logs_dir/master-pod-logs.txt" 2>/dev/null || warn "Could not collect master pod logs"
    kubectl logs -l training.kubeflow.org/job-name="$JOB_NAME",training.kubeflow.org/replica-type=worker > "$latest_logs_dir/worker-pod-logs.txt" 2>/dev/null || warn "Could not collect worker pod logs"

    # Get pod names (pods should be available with cleanPodPolicy: None)
    master_pod=$(kubectl get pods -l training.kubeflow.org/job-name="$JOB_NAME",training.kubeflow.org/replica-type=master -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    # Collect model artifacts from mounted volumes and pod
    log "Collecting model artifacts..."
    
    # Try volume mount first (preferred) - UPDATED for new mnist.py structure
    model_found=false
    
    # Check for new mnist.py model files (priority order)
    if [[ -f "output/mnist_model.pt" ]]; then
        log "Found latest model in mounted volume, moving to models/latest..."
        mv "output/mnist_model.pt" "$latest_models_dir/mnist_model.pt"
        success "Latest model moved: $(ls -lh "$latest_models_dir/mnist_model.pt" | awk '{print $5}')"
        model_found=true
    fi
    
    if [[ -f "output/mnist_model_best.pt" ]]; then
        log "Found best model in mounted volume, moving to models/latest..."
        mv "output/mnist_model_best.pt" "$latest_models_dir/mnist_model_best.pt"
        success "Best model moved: $(ls -lh "$latest_models_dir/mnist_model_best.pt" | awk '{print $5}')"
        model_found=true
    fi
    
    if [[ -f "output/checkpoints/latest_checkpoint.pth" ]]; then
        log "Found latest checkpoint in mounted volume, moving to models/latest..."
        mkdir -p "$latest_models_dir/checkpoints"
        mv "output/checkpoints/latest_checkpoint.pth" "$latest_models_dir/checkpoints/latest_checkpoint.pth"
        success "Latest checkpoint moved: $(ls -lh "$latest_models_dir/checkpoints/latest_checkpoint.pth" | awk '{print $5}')"
        model_found=true
    fi
    
    # Fallback to old file names for compatibility
    if [[ -f "output/trained-model.pth" ]]; then
        log "Found legacy model in mounted volume, moving to models/latest..."
        mv "output/trained-model.pth" "$latest_models_dir/trained-model.pth"
        success "Legacy model moved: $(ls -lh "$latest_models_dir/trained-model.pth" | awk '{print $5}')"
        model_found=true
    fi
    
    # If no models found in volume mount, try pod copy
    if [[ "$model_found" == false ]] && [[ -n "$master_pod" ]]; then
        log "No models found in volume mount, copying from pod: $master_pod"
        if kubectl cp "$master_pod":/output/mnist_model.pt "$latest_models_dir/mnist_model.pt" 2>/dev/null; then
            success "Latest model copied from pod: $(ls -lh "$latest_models_dir/mnist_model.pt" | awk '{print $5}')"
            model_found=true
        elif kubectl cp "$master_pod":/output/mnist_model_best.pt "$latest_models_dir/mnist_model_best.pt" 2>/dev/null; then
            success "Best model copied from pod: $(ls -lh "$latest_models_dir/mnist_model_best.pt" | awk '{print $5}')"
            model_found=true
        elif kubectl cp "$master_pod":/output/checkpoints/latest_checkpoint.pth "$latest_models_dir/checkpoints/latest_checkpoint.pth" 2>/dev/null; then
            success "Latest checkpoint copied from pod: $(ls -lh "$latest_models_dir/checkpoints/latest_checkpoint.pth" | awk '{print $5}')"
            model_found=true
        else
            error "Failed to collect model from both volume mount and pod. Check logs: kubectl logs $master_pod"
        fi
    elif [[ "$model_found" == false ]]; then
        error "No models found in volume mount and no pod available for copying"
    fi
    
    # Collect training metadata
    if [[ -f "output/training_metadata.txt" ]]; then
        mv "output/training_metadata.txt" "$latest_models_dir/training_metadata.txt"
        log "Training metadata moved to models/latest"
    elif [[ -n "$master_pod" ]]; then
        kubectl cp "$master_pod":/output/training_metadata.txt "$latest_models_dir/training_metadata.txt" 2>/dev/null || \
        warn "Could not collect training metadata from pod"
    else
        warn "Training metadata not found in volume mount or pod"
    fi

    # Create job summary
    cat > "$latest_models_dir/job-info.txt" << EOF
Training Job Summary
===================
Job Name: $JOB_NAME
Timestamp: $timestamp
Output Directory: $latest_models_dir
Configuration:
  - Epochs: $EPOCHS
  - Workers: $WORKERS
  - Batch Size: $BATCH_SIZE
  - Learning Rate: $LEARNING_RATE

Status: $(kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
Completion Time: $(date)
EOF

    # Archive previous latest if it exists
    if [[ -d "$models_dir/latest" && "$(ls -A "$models_dir/latest" 2>/dev/null)" ]]; then
        # Check if it's different from current run
        if [[ ! -f "$models_dir/latest/job-info.txt" ]] || [[ "$(grep -c "$timestamp" "$models_dir/latest/job-info.txt" 2>/dev/null)" -eq 0 ]]; then
            log "Archiving previous latest models..."
            mv "$models_dir/latest" "$run_archive_dir/models"
            success "Previous models archived to: $run_archive_dir/models"
        fi
    fi

    # Create simple symlink to latest
    rm -f output/latest
    ln -sf "models/latest" output/latest

    # Clean up any remaining loose files
    if [[ -d "output/checkpoints" && "$(ls -A "output/checkpoints" 2>/dev/null)" ]]; then
        rm -rf "output/checkpoints"
        log "Cleaned up loose checkpoints directory"
    fi

    success "Training artifacts organized in clean structure:"
    echo "   📁 Models: $latest_models_dir"
    echo "   📁 Logs: $latest_logs_dir"
    echo "   📁 Archive: $run_archive_dir"
    echo "   🔗 Latest symlink: output/latest -> models/latest"
}

# Run model inference
run_inference() {
    section "Model Inference Testing"

    # Determine model path
    model_path=""
    if [[ -n "$MODEL_PATH" ]]; then
        # Use user-specified model
        if [[ -f "$MODEL_PATH" ]]; then
            model_path="$MODEL_PATH"
            log "Using specified model: $model_path"
        else
            error "Specified model not found: $MODEL_PATH"
        fi
    else
        # Auto-detect latest model (UPDATED for new organized structure)
        if [[ -f "output/models/latest/mnist_model.pt" ]]; then
            model_path="output/models/latest/mnist_model.pt"
        elif [[ -f "output/models/latest/mnist_model_best.pt" ]]; then
            model_path="output/models/latest/mnist_model_best.pt"
        elif [[ -f "output/models/latest/checkpoints/latest_checkpoint.pth" ]]; then
            model_path="output/models/latest/checkpoints/latest_checkpoint.pth"
        # Fallback to old structure for compatibility
        elif [[ -f "output/mnist_model.pt" ]]; then
            model_path="output/mnist_model.pt"
        elif [[ -f "output/mnist_model_best.pt" ]]; then
            model_path="output/mnist_model_best.pt"
        elif [[ -f "output/checkpoints/latest_checkpoint.pth" ]]; then
            model_path="output/checkpoints/latest_checkpoint.pth"
        else
            # Fallback: Look for most recent model in old structure
                    # Fallback: Look for most recent model in old structure
        latest_dir=$(ls -t output/mnist-training_* 2>/dev/null | head -1 || echo "")
        if [[ -n "$latest_dir" && -f "output/$latest_dir/trained-model.pth" ]]; then
            model_path="output/$latest_dir/trained-model.pth"
        fi
        fi
        
        if [[ -z "$model_path" ]]; then
            error "No trained model found! Run training first with 'make submit-job'"
            echo "Expected model files:"
            echo "  - output/models/latest/mnist_model.pt (latest checkpoint)"
            echo "  - output/models/latest/mnist_model_best.pt (best model)"
            echo "  - output/models/latest/checkpoints/latest_checkpoint.pth (latest checkpoint)"
            echo "  - output/latest/ (symlink to models/latest)"
        fi
        
        log "Auto-detected model: $model_path"
    fi

    success "Using model: $model_path"

    # Handle test images
    if [[ -n "$TEST_IMAGE" ]]; then
        # Single test image
        if [[ -f "$TEST_IMAGE" ]]; then
            log "Testing single image: $TEST_IMAGE"
            echo "📸 Testing $(basename "$TEST_IMAGE"):"
            python scripts/test_mnist_model.py --image "$TEST_IMAGE" --model "$model_path" || warn "Failed to test $TEST_IMAGE"
        else
            error "Test image not found: $TEST_IMAGE"
        fi
    else
        # Use test images directories
        workflow_test_images="examples/01-complete-workflow/test_images"
        root_test_images="test_images"
        
        # Priority order for test images directory selection
        if [[ -d "$workflow_test_images" && $(ls "$workflow_test_images"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) -gt 0 ]]; then
            test_images_dir="$workflow_test_images"
            log "Using existing test images from $test_images_dir"
        elif [[ -d "$root_test_images" && $(ls "$root_test_images"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) -gt 0 ]]; then
            test_images_dir="$root_test_images"
            log "Using test images from $test_images_dir"
        else
            test_images_dir="$root_test_images"
            mkdir -p "$test_images_dir"
            log "Created $test_images_dir/ directory for your test images"
        fi

        # Check if any test images exist
        if [[ $(ls "$test_images_dir"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) -eq 0 ]]; then
            warn "No test images found in $test_images_dir/. Skipping inference testing."
            echo "📸 To test inference, add your own handwritten digit images to $test_images_dir/"
            echo "   Or use: TEST_IMAGE=your_image.png make run-inference"
            return
        fi

        # Test with available images
        log "Testing with images from $test_images_dir/..."
        for img in "$test_images_dir"/*.{jpg,jpeg,png,gif}; do
            if [[ -f "$img" ]]; then
                echo "📸 Testing $(basename "$img"):"
                python scripts/test_mnist_model.py --image "$img" --model "$model_path" || warn "Failed to test $img"
                echo
            fi
        done

        # Test batch processing
        log "Running batch inference..."
        echo "📦 Batch processing all test images:"
        output_dir=$(dirname "$model_path")
        python scripts/test_mnist_model.py --batch "$test_images_dir/" --model "$model_path" > "$output_dir/inference-results.txt" || warn "Batch processing failed"

        # Display batch results
        if [[ -f "$output_dir/inference-results.txt" ]]; then
            echo "📊 Batch Results Summary:"
            tail -10 "$output_dir/inference-results.txt" || true
        fi
    fi

    success "Model inference completed"
}

# Show training results and summary
show_results() {
    section "Training Results Summary"

    # Find latest output directory
    output_dir=""
    if [[ -L "output/latest" ]]; then
        output_dir="output/$(readlink output/latest)"
    else
        # Fallback: Look for most recent model in old structure
        latest_dir=$(ls -t output/mnist-training_* 2>/dev/null | head -1 || echo "")
        if [[ -n "$latest_dir" ]]; then
            output_dir="output/$latest_dir"
        fi
    fi

    if [[ -z "$output_dir" || ! -d "$output_dir" ]]; then
        error "No training results found! Run training first with 'make submit-job'"
    fi

    echo "🎯 Complete Workflow Results:"
    echo "============================="
    echo
    echo "📁 Training Artifacts:"
    echo "  📍 Location: $output_dir"
    if [[ -f "$output_dir/trained-model.pth" ]]; then
        echo "  🏷️  Model: $(ls -lh "$output_dir/trained-model.pth" | awk '{print $5}') trained-model.pth"
    fi
    echo "  📋 Logs: master-pod-logs.txt, worker-pod-logs.txt"
    echo "  📊 Metadata: training_metadata.txt"
    echo

    echo "🔍 Inference Results:"
    # Use same logic to find test images directory
    workflow_test_images="examples/01-complete-workflow/test_images"
    root_test_images="test_images"
    
    if [[ -d "$workflow_test_images" && $(ls "$workflow_test_images"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) -gt 0 ]]; then
        test_images_dir="$workflow_test_images"
    elif [[ -d "$root_test_images" && $(ls "$root_test_images"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) -gt 0 ]]; then
        test_images_dir="$root_test_images"
    else
        test_images_dir="$root_test_images"
    fi
    
    if [[ -d "$test_images_dir" ]]; then
        echo "  📸 Test Images: $(ls "$test_images_dir"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) images tested from $test_images_dir"
    fi
    if [[ -f "$output_dir/inference-results.txt" ]]; then
        echo "  📦 Batch Results: $output_dir/inference-results.txt"
    fi
    echo

    echo "📈 Training Summary:"
    if [[ -f "$output_dir/training_metadata.txt" ]]; then
        grep -E "(Final|Test) Accuracy|Loss" "$output_dir/training_metadata.txt" 2>/dev/null || echo "  Training metrics saved to training_metadata.txt"
    else
        echo "  Training completed successfully (see logs for details)"
    fi
    echo

    echo "🔗 Next Steps:"
    echo "  1. Review training logs: cat $output_dir/master-pod-logs.txt"
    if [[ -f "$output_dir/trained-model.pth" ]]; then
        echo "  2. Test with your images: TEST_IMAGE=your_image.png make run-inference"
    fi
    echo "  3. Try other examples: ls examples/"
    echo "  4. Scale up training: edit configs/pytorch-distributed-job.yaml (increase workers)"
    echo

    success "🎉 Workflow results displayed!"
}

# Debug training job
debug_training() {
    section "Training Debug Information"
    
    log "Checking cluster status..."
    echo "Cluster info:"
    kubectl cluster-info || warn "Cluster not accessible"
    echo
    
    log "Checking training operator..."
    echo "Training operator status:"
    kubectl get deployment -n kubeflow training-operator 2>/dev/null || warn "Training operator not found"
    echo
    
    log "Checking PyTorchJob status..."
    if kubectl get pytorchjob "$JOB_NAME" &>/dev/null; then
        echo "PyTorchJob status:"
        kubectl get pytorchjob "$JOB_NAME" -o yaml
        echo
        
        echo "PyTorchJob conditions:"
        kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.status.conditions[*]}' | jq '.' 2>/dev/null || \
        kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.status.conditions[*]}'
        echo
    else
        warn "PyTorchJob '$JOB_NAME' not found"
    fi
    
    log "Checking pods..."
    if kubectl get pods -l training.kubeflow.org/job-name="$JOB_NAME" &>/dev/null; then
        echo "Pod status:"
        kubectl get pods -l training.kubeflow.org/job-name="$JOB_NAME" -o wide
        echo
        
        echo "Pod descriptions:"
        kubectl describe pods -l training.kubeflow.org/job-name="$JOB_NAME"
        echo
        
        echo "Pod logs:"
        for pod in $(kubectl get pods -l training.kubeflow.org/job-name="$JOB_NAME" -o name); do
            echo "--- Logs for $pod ---"
            kubectl logs "$pod" --tail=50 || warn "Could not get logs for $pod"
            echo
        done
    else
        warn "No pods found for job '$JOB_NAME'"
    fi
    
    log "Checking output directory..."
    echo "Output directory contents:"
    ls -la output/ 2>/dev/null || warn "output/ directory not found"
    echo
    
    log "Checking training script..."
    echo "Training script location:"
    ls -la scripts/mnist.py 2>/dev/null || warn "Training script not found"
    echo
    
    log "Checking PyTorchJob configuration..."
    echo "PyTorchJob YAML:"
    cat configs/pytorch-distributed-job.yaml
    echo
    
    success "Debug information collected"
    echo
    echo "🎯 Common issues to check:"
    echo "  • Pod failed to start: Check pod descriptions above"
    echo "  • Training script error: Check pod logs above"
    echo "  • Model not saved: Check if /output directory is writable in pod"
    echo "  • Volume mount issues: Check Kind cluster extraMounts configuration"
    echo "  • Training operator issues: Check operator deployment status"
    echo
    echo "🔧 Debugging commands:"
    echo "  • Watch job status: kubectl get pytorchjob $JOB_NAME -w"
    echo "  • Watch pod status: kubectl get pods -l training.kubeflow.org/job-name=$JOB_NAME -w"
    echo "  • Stream logs: kubectl logs -f <pod-name>"
    echo "  • Exec into pod: kubectl exec -it <pod-name> -- /bin/bash"
    echo "  • Check events: kubectl get events --sort-by=.metadata.creationTimestamp"
}

# Run complete workflow
run_complete_workflow() {
    section "Complete ML Workflow: Training + Inference"
    echo "Configuration:"
    echo "  📊 Epochs: $EPOCHS"
    echo "  👥 Workers: $WORKERS" 
    echo "  📦 Batch Size: $BATCH_SIZE"
    echo "  📈 Learning Rate: $LEARNING_RATE"
    echo "  ⏱️  Timeout: $JOB_TIMEOUT seconds"
    echo
    
    # Phase 1: Infrastructure Setup
    log "Phase 1: Infrastructure Setup"
    if ! kubectl cluster-info &> /dev/null; then
        log "No cluster found, creating Kind cluster..."
        create_cluster_only
    else
        success "Cluster already available"
    fi
    
    # Phase 2: Training Operator Installation
    log "Phase 2: Training Operator Installation"
    install_operator_only
    
    # Phase 3: Distributed Training
    submit_and_monitor_job
    monitor_job
    collect_artifacts
    
    # Phase 4: Model Inference
    run_inference
    
    # Phase 5: Results Summary
    show_results
    
    success "🎉 Complete workflow finished successfully!"
}

# Handle script arguments
case "${1:-}" in
    "")
        main
        ;;
    "install-deps")
        install_deps
        ;;
    "cluster-only")
        create_cluster_only
        ;;
    "use-existing")
        use_existing_cluster
        ;;
    "install-operator")
        install_operator_only
        ;;
    "prepare-training")
        prepare_training_env
        ;;
    "setup-training")
        setup_training
        ;;
    "check-requirements")
        check_requirements
        ;;
    "verify-system")
        verify_system
        ;;
    "validate")
        validate_setup
        ;;
    "download-mnist")
        install_python_deps
        download_mnist
        ;;
    "cluster-info")
        print_cluster_info
        ;;
    "submit-job")
        submit_and_monitor_job
        ;;
    "monitor-job")
        monitor_job
        ;;
    "collect-artifacts")
        collect_artifacts
        ;;
    "run-inference")
        run_inference
        ;;
    "show-results")
        show_results
        ;;
    "debug-training")
        debug_training
        ;;
    "run-workflow")
        run_complete_workflow
        ;;
    *)
        echo "Usage: $0 [command]"
        echo ""
        echo "Infrastructure Commands:"
        echo "  (none)              - Full infrastructure setup (cluster + dependencies)"
        echo "  install-deps        - Install only dependencies (Docker/Podman, Python, kubectl, kind)"
        echo "  cluster-only        - Create cluster (prompt for existing vs new)"
        echo "  use-existing        - Use existing cluster (skip cluster creation)"
        echo "  check-requirements  - Check system requirements only"
        echo "  verify-system       - Comprehensive system and dependency verification"
        echo ""
        echo "Training Commands:"
        echo "  install-operator    - Install Kubeflow training operator only"
        echo "  prepare-training    - Prepare training environment (ConfigMap + dataset)"
        echo "  setup-training      - Complete training setup (operator + environment)"
        echo "  download-mnist      - Download MNIST dataset only"
        echo ""
        echo "Workflow Commands:"
        echo "  submit-job          - Submit and monitor distributed training job"
        echo "  monitor-job         - Monitor existing training job progress"
        echo "  collect-artifacts   - Collect training artifacts (model, logs, metadata)"
        echo "  run-inference       - Run model inference on test images"
        echo "  show-results        - Display comprehensive training results summary"
        echo "  debug-training      - Show detailed debug information for training issues"
        echo "  run-workflow        - Run complete end-to-end workflow (all phases)"
        echo ""
        echo "Utility Commands:"
        echo "  validate            - Validate existing cluster setup"
        echo "  cluster-info        - Show cluster information"
        echo ""
        echo "Examples:"
        echo "  make setup                    # Full setup with prompts"
echo "  make verify-system            # Check all dependencies and readiness"
echo "  make use-existing             # Use existing cluster"
echo "  make install-operator         # Install operator on existing cluster"
        exit 1
        ;;
esac 