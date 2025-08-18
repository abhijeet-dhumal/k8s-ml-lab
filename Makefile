# Variables
JOB_NAME := mnist-training
CLUSTER_NAME := pytorch-training-cluster
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

# Common setup script execution
SETUP_SCRIPT = @chmod +x bin/setup.sh && ./bin/setup.sh

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Distributed PyTorch Training Setup"
	@echo "=================================="
	@echo ""
	@echo "$(BLUE)Ideal Flow (Manual Step-by-Step):$(NC)"
	@echo "======================================"
	@echo "1. $(GREEN)setup$(NC)                    - Full infrastructure setup"
	@echo "2. $(GREEN)submit-job$(NC)               - Submit training job"
	@echo "3. $(GREEN)status$(NC)                   - Monitor job progress"
	@echo "4. $(GREEN)logs$(NC)                     - View training logs (optional)"
	@echo "5. $(GREEN)inference$(NC)                - Run model inference"
	@echo "6. $(GREEN)cleanup$(NC)                  - Clean up resources"
	@echo ""
	@echo "Or use: $(GREEN)run-e2e-workflow$(NC)   - Run all steps automatically"
	@echo ""
	@echo "$(BLUE)Available commands:$(NC)"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[32m%-20s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# SETUP & INSTALLATION
# ==============================================================================

.PHONY: setup
setup: ## Full infrastructure setup (cluster + dependencies + training env)
	@echo "$(BLUE)Setting up distributed PyTorch training environment...$(NC)"
	$(SETUP_SCRIPT)

.PHONY: verify-system
verify-system: ## Comprehensive system and dependency verification
	@echo "$(BLUE)Running comprehensive system verification...$(NC)"
	$(SETUP_SCRIPT) verify-system

.PHONY: use-existing
use-existing: ## Use existing cluster (skip cluster creation)
	@echo "$(BLUE)Configuring existing cluster...$(NC)"
	$(SETUP_SCRIPT) use-existing

# ==============================================================================
# TRAINING & WORKFLOWS
# ==============================================================================

.PHONY: submit-job
submit-job: ## Submit PyTorch distributed training job
	@echo "$(BLUE)Submitting PyTorch training job...$(NC)"
	$(SETUP_SCRIPT) submit-job

.PHONY: submit-single-job
submit-single-job: ## Submit simple single-pod training job
	@echo "$(BLUE)Submitting simple single-pod training job...$(NC)"
	@kubectl apply -f configs/simple-single-pod-job.yaml
	@echo "$(GREEN)✓ Single-pod job submitted$(NC)"

.PHONY: run-e2e-workflow
run-e2e-workflow: ## Run complete end-to-end workflow (training + inference + results)
	@echo "$(BLUE)Running complete end-to-end workflow...$(NC)"
	$(SETUP_SCRIPT) run-workflow

.PHONY: status
status: ## Show job status, pods, and recent events
	@echo "$(BLUE)Job Status:$(NC)"
	@echo "==========="
	@echo "Distributed Training Job:"
	@kubectl get pytorchjob $(JOB_NAME) -o wide || echo "Job not found"
	@echo ""
	@echo "Single Training Job:"
	@kubectl get job simple-mnist-training -o wide || echo "Job not found"
	@echo ""
	@echo "All Pods:"
	@kubectl get pods -l training.kubeflow.org/job-name=$(JOB_NAME) -o wide || echo "No distributed pods found"
	@kubectl get pods -l job-name=simple-mnist-training -o wide || echo "No single job pods found"
	@echo ""
	@echo "Recent Events:"
	@kubectl get events --field-selector involvedObject.name=$(JOB_NAME) --sort-by='.lastTimestamp' | tail -5 || echo "No distributed job events found"
	@kubectl get events --field-selector involvedObject.name=simple-mnist-training --sort-by='.lastTimestamp' | tail -5 || echo "No single job events found"

.PHONY: logs
logs: ## View logs from master pod (real-time)
	@echo "$(BLUE)Master Pod Logs:$(NC)"
	@kubectl logs -l training.kubeflow.org/job-name=$(JOB_NAME),training.kubeflow.org/replica-type=master -f --tail=100

.PHONY: debug
debug: ## Show comprehensive debugging information
	@echo "$(BLUE)Comprehensive Debugging Information:$(NC)"
	@echo "===================================="
	@echo "Cluster Context:"
	@kubectl config current-context || echo "No context found"
	@echo ""
	@echo "Cluster Info:"
	@kubectl cluster-info || echo "Cluster not accessible"
	@echo ""
	@echo "Nodes:"
	@kubectl get nodes -o wide || echo "No nodes found"
	@echo ""
	@echo "PyTorchJob CRD:"
	@kubectl get crd pytorchjobs.kubeflow.org || echo "PyTorchJob CRD not found"
	@echo ""
	@echo "Training Operator:"
	@kubectl get deployment training-operator -n kubeflow || echo "Training operator not found"
	@echo ""
	@echo "Current Job:"
	@kubectl get pytorchjob $(JOB_NAME) || echo "Job not found"
	@echo ""
	@echo "Job Pods:"
	@kubectl get pods -l training.kubeflow.org/job-name=$(JOB_NAME) -o wide || echo "No pods found"

.PHONY: restart
restart: ## Restart training job (delete + submit)
	@echo "$(YELLOW)Restarting training job...$(NC)"
	@kubectl delete pytorchjob $(JOB_NAME) || echo "Job not found"
	@sleep 5
	$(SETUP_SCRIPT) submit-job

.PHONY: inference
inference: ## Run model inference on test images (TEST_IMAGE=path or TEST_IMAGES_DIR=path)
	@echo "$(BLUE)Running model inference...$(NC)"
	$(SETUP_SCRIPT) run-inference

# ==============================================================================
# CLEANUP
# ==============================================================================

.PHONY: cleanup
cleanup: ## Clean up jobs and resources (keep cluster)
	@echo "$(YELLOW)Cleaning up resources...$(NC)"
	@kubectl delete pytorchjob $(JOB_NAME) || echo "Job not found"
	@kubectl delete job simple-mnist-training || echo "Single job not found"
	@kubectl delete configmap pytorch-training-script || echo "ConfigMap not found"
	@echo "$(GREEN)✓ Resources cleaned up$(NC)"

.PHONY: cleanup-all
cleanup-all: cleanup ## Delete entire Kind cluster and all resources
	@echo -e "$(YELLOW)Deleting Kind cluster...$(NC)"
	@kind delete cluster --name $(CLUSTER_NAME) || echo "Cluster not found"
	@echo -e "$(GREEN)✓ Complete cleanup done$(NC)"

# ==============================================================================
# ALIASES FOR COMPATIBILITY
# ==============================================================================

.PHONY: check-requirements
check-requirements: verify-system ## Alias for verify-system

.PHONY: install-operator
install-operator: ## Install Kubeflow training operator (standalone)
	@echo -e "$(BLUE)Installing Kubeflow training operator...$(NC)"
	$(SETUP_SCRIPT) install-operator

# Make bin/setup.sh executable
bin/setup.sh:
	@chmod +x bin/setup.sh

# Default target when no argument is given
.DEFAULT_GOAL := help 