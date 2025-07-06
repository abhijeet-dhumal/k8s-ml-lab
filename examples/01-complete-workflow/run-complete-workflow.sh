#!/bin/bash

# Complete Workflow: Distributed Training + Model Inference
# Usage: ./run-complete-workflow.sh [phase]
# Phases: setup, training, inference, results, all
# Example: ./run-complete-workflow.sh setup

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration
EPOCHS=${EPOCHS:-5}
WORKERS=${WORKERS:-2}
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-0.001}
JOB_TIMEOUT=${JOB_TIMEOUT:-600}  # 10 minutes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
    exit 1
}

section() {
    echo
    echo -e "${CYAN}🚀 $1${NC}"
    echo -e "${CYAN}$(printf '=%.0s' {1..50})${NC}"
}

show_usage() {
    echo "Usage: $0 [phase]"
    echo
    echo "Phases:"
    echo "  setup                    - Set up infrastructure (Kind cluster + dependencies)"
    echo "  install-training-operator - Install Kubeflow training operator"
    echo "  training                 - Run distributed training job"
    echo "  inference                - Test trained model with sample images"
    echo "  results                  - Show training results and next steps"
    echo "  debug                    - Debug training issues"
    echo "  all                      - Run complete workflow (all phases)"
    echo
    echo "Environment variables:"
    echo "  EPOCHS=${EPOCHS}      - Number of training epochs"
    echo "  WORKERS=${WORKERS}     - Number of worker nodes"
    echo "  BATCH_SIZE=${BATCH_SIZE}   - Training batch size"
    echo "  LEARNING_RATE=${LEARNING_RATE} - Learning rate"
    echo "  JOB_TIMEOUT=${JOB_TIMEOUT}   - Job timeout in seconds"
    echo
    echo "Inference-specific variables:"
    echo "  TEST_IMAGE=path/to/image.jpg     - Single test image"
    echo "  TEST_IMAGES_DIR=path/to/dir/     - Directory of test images (default: test_images/)"
    echo "  MODEL_PATH=path/to/model.pth     - Specific model to use"
    echo
    echo "Examples:"
    echo "  $0 setup                             # Setup cluster and dependencies"
    echo "  $0 install-training-operator         # Install training operator"
    echo "  $0 training                          # Run training only"
    echo "  EPOCHS=10 $0 training                # Training with 10 epochs"
    echo "  $0 inference                         # Test model inference (uses test_images/ by default)"
    echo "  TEST_IMAGE=my_digit.jpg $0 inference # Test with custom image"
    echo "  TEST_IMAGES_DIR=my_images/ $0 inference # Test with custom directory"
    echo "  MODEL_PATH=output/old_model.pth $0 inference # Use specific model"
    echo "  $0 all                               # Complete workflow (all phases)"
}

# ==============================================================================
# PHASE 1: INFRASTRUCTURE SETUP
# ==============================================================================
run_setup() {
    section "Phase 1: Infrastructure Setup"

    log "Checking dependencies..."
    
    # Check container runtime (Docker or Podman)
    if ! command -v docker &> /dev/null && ! command -v podman &> /dev/null; then
        error "Container runtime not found. Please install Docker or Podman: ./setup.sh install-deps"
    fi
    
    # Check other dependencies
    for cmd in kubectl kind python; do
        if ! command -v "$cmd" &> /dev/null; then
            error "$cmd is not installed. Please run: ./setup.sh install-deps"
        fi
    done
    success "All dependencies available"

    log "Setting up training environment..."
    if ! kubectl cluster-info &> /dev/null; then
        log "No cluster found, creating Kind cluster..."
        ./setup.sh cluster-only
    else
        success "Cluster already available"
    fi

    success "Infrastructure ready"
    echo
    echo "🎯 Next Steps:"
    echo "  • Install operator: $0 install-training-operator"
    echo "  • Check cluster: kubectl get nodes"
    echo "  • View cluster info: kubectl cluster-info"
}

# ==============================================================================
# PHASE 2: TRAINING OPERATOR INSTALLATION
# ==============================================================================
run_operator() {
    section "Phase 2: Training Operator Installation"

    log "Installing training operator..."
    ./setup.sh install-operator

    success "Training operator installed"
    echo
    echo "🎯 Next Steps:"
    echo "  • Run training: $0 training"
    echo "  • View operator: kubectl get pods -n kubeflow"
    echo "  • Check operator status: kubectl get deployment -n kubeflow"
}

# ==============================================================================
# PHASE 3: DISTRIBUTED TRAINING
# ==============================================================================
run_training() {
    section "Phase 3: Distributed Training"

    echo "Training Configuration:"
    echo "  📊 Epochs: $EPOCHS"
    echo "  👥 Workers: $WORKERS" 
    echo "  📦 Batch Size: $BATCH_SIZE"
    echo "  📈 Learning Rate: $LEARNING_RATE"
    echo "  ⏱️  Timeout: $JOB_TIMEOUT seconds"
    echo

    JOB_NAME="pytorch-single-worker-distributed"

    log "Cleaning up any existing jobs..."
    kubectl delete pytorchjob "$JOB_NAME" --ignore-not-found=true
    sleep 5

    log "Preparing required directories..."
    mkdir -p input output scripts
    
    # Ensure directories exist for Kind volume mounts
    log "Checking required directories for Kind volume mounts..."
    if [[ ! -d "scripts" ]]; then
        error "scripts/ directory not found. Required for Kind volume mount."
    fi
    if [[ ! -d "input" ]]; then
        warning "input/ directory not found. Creating it now..."
        mkdir -p input
    fi
    if [[ ! -d "output" ]]; then
        warning "output/ directory not found. Creating it now..."
        mkdir -p output
    fi
    
    # Check if training script exists
    if [[ ! -f "scripts/distributed_mnist_training.py" ]]; then
        error "Training script not found: scripts/distributed_mnist_training.py"
    fi
    
    # Download MNIST dataset if not already present
    if [[ ! -d "input/MNIST" ]]; then
        log "Downloading MNIST dataset..."
        # Use existing download_mnist function from setup.sh
        ./setup.sh download-mnist
    else
        log "MNIST dataset already exists in input/"
    fi
    
    log "All required directories ready:"
    ls -la scripts/ input/ output/ | head -10

    log "Submitting distributed training job..."
    kubectl apply -f configs/pytorch-distributed-job.yaml

    # Wait for job completion with progress monitoring
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
                    warning "PyTorchJob succeeded but training script may have issues. Check logs."
                fi
                
                # Check if model was saved
                if kubectl exec "$master_pod" -- test -f /output/trained-model.pth 2>/dev/null; then
                    success "Model file found in pod!"
                else
                    warning "Model file not found in pod - training may not have saved properly"
                fi
            else
                warning "Cannot verify training success - master pod not found"
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
                warning "Failed pods detected: $failing_pods"
            elif [[ -n "$pending_pods" ]]; then
                warning "Pods still pending: $pending_pods"
            fi
        else
            # Simple progress dots
            echo -n "."
        fi
        
        sleep 15
        counter=$((counter + 15))
    done

    # Collect training artifacts
    log "Collecting training artifacts..."
    timestamp=$(date '+%Y-%m-%d_%H-%M-%S')
    output_dir="output/${JOB_NAME}_${timestamp}"
    mkdir -p "$output_dir"
    
    # Brief final status
    log "Final training status:"
    kubectl get pytorchjob "$JOB_NAME" -o custom-columns=NAME:.metadata.name,STATE:.status.conditions[-1].type,AGE:.metadata.creationTimestamp --no-headers 2>/dev/null || true

    # Collect logs from pods
    log "Collecting pod logs..."
    kubectl logs -l training.kubeflow.org/job-name="$JOB_NAME",training.kubeflow.org/replica-type=master > "$output_dir/master-pod-logs.txt" 2>/dev/null || warning "Could not collect master pod logs"
    kubectl logs -l training.kubeflow.org/job-name="$JOB_NAME",training.kubeflow.org/replica-type=worker > "$output_dir/worker-pod-logs.txt" 2>/dev/null || warning "Could not collect worker pod logs"

    # Get pod names (pods should be available with cleanPodPolicy: None)
    master_pod=$(kubectl get pods -l training.kubeflow.org/job-name="$JOB_NAME",training.kubeflow.org/replica-type=master -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    
    # Collect model artifacts from mounted volumes and pod
    log "Collecting model artifacts..."
    
    # Try volume mount first (preferred)
    if [[ -f "output/trained-model.pth" ]]; then
        log "Found model in mounted volume, moving to job directory..."
        mv "output/trained-model.pth" "$output_dir/trained-model.pth"
        success "Model moved from volume mount: $(ls -lh "$output_dir/trained-model.pth" | awk '{print $5}')"
    elif [[ -n "$master_pod" ]]; then
        # Fallback to pod copy if volume mount fails
        log "Copying model from pod: $master_pod"
        if kubectl cp "$master_pod":/output/trained-model.pth "$output_dir/trained-model.pth" 2>/dev/null; then
            success "Model copied from pod: $(ls -lh "$output_dir/trained-model.pth" | awk '{print $5}')"
        else
            error "Failed to collect model from both volume mount and pod. Check logs: kubectl logs $master_pod"
        fi
    else
        error "No model found in volume mount and no pod available for copying"
        warning "Check job logs for details. Model may not have been saved properly."
    fi
    
    # Collect training metadata
    if [[ -f "output/training_metadata.txt" ]]; then
        mv "output/training_metadata.txt" "$output_dir/training_metadata.txt"
        log "Training metadata moved from mounted volume"
    elif [[ -n "$master_pod" ]]; then
        kubectl cp "$master_pod":/output/training_metadata.txt "$output_dir/training_metadata.txt" 2>/dev/null || \
        warning "Could not collect training metadata from pod"
    else
        warning "Training metadata not found in volume mount or pod"
    fi

    # Create job summary
    cat > "$output_dir/job-info.txt" << EOF
Training Job Summary
===================
Job Name: $JOB_NAME
Timestamp: $timestamp
Output Directory: $output_dir
Configuration:
  - Epochs: $EPOCHS
  - Workers: $WORKERS
  - Batch Size: $BATCH_SIZE
  - Learning Rate: $LEARNING_RATE

Status: $(kubectl get pytorchjob "$JOB_NAME" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
Completion Time: $(date)
EOF

    # Create symlink to latest
    rm -f output/latest
    ln -sf "${JOB_NAME}_${timestamp}" output/latest

    success "Training artifacts saved to: $output_dir"
    echo
    echo "🎯 Next Steps:"
    echo "  • Test model: $0 inference"
    echo "  • View logs: cat $output_dir/master-pod-logs.txt"
    echo "  • Check model: ls -la $output_dir/trained-model.pth"
}

# ==============================================================================
# PHASE 4: MODEL INFERENCE
# ==============================================================================
run_inference() {
    section "Phase 4: Model Inference"

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
        # Auto-detect latest model
        if [[ -f "output/latest/trained-model.pth" ]]; then
            model_path="output/latest/trained-model.pth"
        else
            # Look for most recent model
            latest_dir=$(ls -t output/pytorch-single-worker-distributed_* 2>/dev/null | head -1 || echo "")
            if [[ -n "$latest_dir" && -f "output/$latest_dir/trained-model.pth" ]]; then
                model_path="output/$latest_dir/trained-model.pth"
            fi
        fi
        
        if [[ -z "$model_path" ]]; then
            error "No trained model found! Run training first: $0 training"
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
            python scripts/test_mnist_model.py --image "$TEST_IMAGE" --model "$model_path" || warning "Failed to test $TEST_IMAGE"
            echo
        else
            error "Test image not found: $TEST_IMAGE"
        fi
    elif [[ -n "$TEST_IMAGES_DIR" ]]; then
        # User-specified directory
        if [[ -d "$TEST_IMAGES_DIR" ]]; then
            log "Testing images from directory: $TEST_IMAGES_DIR"
            
            # Test individual images
            echo "📸 Testing individual images:"
            for img in "$TEST_IMAGES_DIR"/*.{jpg,jpeg,png,gif}; do
                if [[ -f "$img" ]]; then
                    echo "📸 Testing $(basename "$img"):"
                    python scripts/test_mnist_model.py --image "$img" --model "$model_path" || warning "Failed to test $img"
                    echo
                fi
            done
            
            # Test batch processing
            log "Running batch inference..."
            echo "📦 Batch processing all test images:"
            output_dir=$(dirname "$model_path")
            python scripts/test_mnist_model.py --batch "$TEST_IMAGES_DIR/" --model "$model_path" > "$output_dir/inference-results.txt" || warning "Batch processing failed"
            
            # Display batch results
            if [[ -f "$output_dir/inference-results.txt" ]]; then
                echo "📊 Batch Results Summary:"
                tail -10 "$output_dir/inference-results.txt" || true
            fi
        else
            error "Test images directory not found: $TEST_IMAGES_DIR"
        fi
    else
        # Improved logic: Use existing test images directory directly
        workflow_test_images="examples/01-complete-workflow/test_images"
        root_test_images="test_images"
        
        # Priority order for test images directory selection
        if [[ -d "$workflow_test_images" && $(ls "$workflow_test_images"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) -gt 0 ]]; then
            # Use existing workflow test images directly
            test_images_dir="$workflow_test_images"
            log "Using existing test images from $test_images_dir"
        elif [[ -d "$root_test_images" && $(ls "$root_test_images"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) -gt 0 ]]; then
            # Use root test images if they exist
            test_images_dir="$root_test_images"
            log "Using test images from $test_images_dir"
        else
            # Create root test images directory as fallback for user images
            test_images_dir="$root_test_images"
            mkdir -p "$test_images_dir"
            log "Created $test_images_dir/ directory for your test images"
        fi

        # Check if any test images exist
        if [[ $(ls "$test_images_dir"/*.{jpg,jpeg,png,gif} 2>/dev/null | wc -l) -eq 0 ]]; then
            warning "No test images found in $test_images_dir/. Skipping inference testing."
            echo "📸 To test inference, add your own handwritten digit images to $test_images_dir/"
            echo "   Or use: TEST_IMAGE=your_image.png $0 inference"
            return
        fi

        # Test with available images
        log "Testing with images from $test_images_dir/..."
        for img in "$test_images_dir"/*.{jpg,jpeg,png,gif}; do
            if [[ -f "$img" ]]; then
                echo "📸 Testing $(basename "$img"):"
                python scripts/test_mnist_model.py --image "$img" --model "$model_path" || warning "Failed to test $img"
                echo
            fi
        done

        # Test batch processing
        log "Running batch inference..."
        echo "📦 Batch processing all test images:"
        output_dir=$(dirname "$model_path")
        python scripts/test_mnist_model.py --batch "$test_images_dir/" --model "$model_path" > "$output_dir/inference-results.txt" || warning "Batch processing failed"

        # Display batch results
        if [[ -f "$output_dir/inference-results.txt" ]]; then
            echo "📊 Batch Results Summary:"
            tail -10 "$output_dir/inference-results.txt" || true
        fi
    fi

    success "Model inference completed"
    echo
    echo "🎯 Next Steps:"
    echo "  • View results: $0 results"
    echo "  • Test more images: TEST_IMAGE=your_image.png $0 inference"
    echo "  • Test directory: TEST_IMAGES_DIR=your_images/ $0 inference"
    if [[ -f "$(dirname "$model_path")/inference-results.txt" ]]; then
        echo "  • View batch results: cat $(dirname "$model_path")/inference-results.txt"
    fi
}

# ==============================================================================
# PHASE 5: RESULTS & SUMMARY
# ==============================================================================
run_results() {
    section "Phase 5: Results Summary"

    # Find latest output directory
    output_dir=""
    if [[ -L "output/latest" ]]; then
        output_dir="output/$(readlink output/latest)"
    else
        latest_dir=$(ls -t output/pytorch-single-worker-distributed_* 2>/dev/null | head -1 || echo "")
        if [[ -n "$latest_dir" ]]; then
            output_dir="output/$latest_dir"
        fi
    fi

    if [[ -z "$output_dir" || ! -d "$output_dir" ]]; then
        error "No training results found! Run training first: $0 training"
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
        echo "  2. Test with your images: python scripts/test_mnist_model.py --image your_image.png --model $output_dir/trained-model.pth"
    fi
    echo "  3. Try other examples: ls examples/"
    echo "  4. Scale up training: edit configs/pytorch-distributed-job.yaml (increase workers)"
    echo

    success "🎉 Workflow results displayed!"
    echo

    # Optional cleanup prompt
    echo "🧹 Cleanup options:"
    echo "  • Keep everything: Job running, model saved, ready for more inference"
    echo "  • Clean job only: ./cleanup.sh (keeps model and cluster)"
    echo "  • Clean everything: make cleanup-all (removes cluster too)"
    echo 
}

# ==============================================================================
# DEBUG FUNCTION
# ==============================================================================
run_debug() {
    section "Training Debug Information"
    
    JOB_NAME="pytorch-single-worker-distributed"
    
    log "Checking cluster status..."
    echo "Cluster info:"
    kubectl cluster-info || warning "Cluster not accessible"
    echo
    
    log "Checking training operator..."
    echo "Training operator status:"
    kubectl get deployment -n kubeflow training-operator 2>/dev/null || warning "Training operator not found"
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
        warning "PyTorchJob '$JOB_NAME' not found"
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
            kubectl logs "$pod" --tail=50 || warning "Could not get logs for $pod"
            echo
        done
    else
        warning "No pods found for job '$JOB_NAME'"
    fi
    
    log "Checking output directory..."
    echo "Output directory contents:"
    ls -la output/ 2>/dev/null || warning "output/ directory not found"
    echo
    
    log "Checking mounted volumes..."
    echo "Host directory contents:"
    ls -la /tmp/kind-data/ 2>/dev/null || warning "Kind data directory not found"
    echo
    
    log "Checking training script..."
    echo "Training script location:"
    ls -la scripts/distributed_mnist_training.py 2>/dev/null || warning "Training script not found"
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

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

# Change to repo root
cd "$REPO_ROOT"

# Parse command line arguments
PHASE="${1:-help}"

case "$PHASE" in
    "setup")
        run_setup
        ;;
    "install-training-operator")
        run_operator
        ;;
    "training")
        run_training
        ;;
    "inference")
        run_inference
        ;;
    "results")
        run_results
        ;;
    "debug")
        run_debug
        ;;
    "all")
        section "Complete ML Workflow: Training + Inference"
        echo "Configuration:"
        echo "  📊 Epochs: $EPOCHS"
        echo "  👥 Workers: $WORKERS" 
        echo "  📦 Batch Size: $BATCH_SIZE"
        echo "  📈 Learning Rate: $LEARNING_RATE"
        echo "  ⏱️  Timeout: $JOB_TIMEOUT seconds"
        echo
        
        run_setup
        run_operator
        run_training
        run_inference
        run_results
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo
        show_usage
        exit 1
        ;;
esac 