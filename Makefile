# Distributed PyTorch Training Setup Makefile
# ==========================================

# Configuration
CLUSTER_NAME = pytorch-training-cluster
JOB_NAME = pytorch-single-worker-distributed
NAMESPACE = default
KUBECONFIG = ~/.kube/config

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Default target
.PHONY: help
help: ## Show this help message
	@echo "Distributed PyTorch Training Setup"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ==============================================================================
# SETUP & INSTALLATION
# ==============================================================================

.PHONY: setup
setup: ## Install dependencies and create cluster
	@echo -e "$(BLUE)Setting up distributed PyTorch training environment...$(NC)"
	@chmod +x setup.sh
	@./setup.sh

.PHONY: install-deps
install-deps: ## Install only dependencies (Docker, Python, kubectl, kind, packages)
	@echo -e "$(BLUE)Installing dependencies...$(NC)"
	@chmod +x setup.sh
	@./setup.sh install-deps

.PHONY: cluster-only
cluster-only: ## Create cluster only (prompt for existing vs new)
	@echo -e "$(BLUE)Creating cluster...$(NC)"
	@chmod +x setup.sh
	@./setup.sh cluster-only

.PHONY: install-operator
install-operator: ## Install Kubeflow training operator only
	@echo -e "$(BLUE)Installing Kubeflow training operator...$(NC)"
	@chmod +x setup.sh
	@./setup.sh install-operator

# ==============================================================================
# CLUSTER MANAGEMENT
# ==============================================================================

.PHONY: cluster-info
cluster-info: ## Show cluster information
	@echo -e "$(BLUE)Cluster Information:$(NC)"
	@echo "==================="
	@kubectl cluster-info
	@echo ""
	@echo "Nodes:"
	@kubectl get nodes -o wide
	@echo ""
	@echo "Training Operator:"
	@kubectl get deployment training-operator -n kubeflow || echo "Training operator not found"

# ==============================================================================
# JOB MANAGEMENT
# ==============================================================================

.PHONY: submit-job
submit-job: ## Submit PyTorch distributed training job
	@echo -e "$(BLUE)Submitting PyTorch training job...$(NC)"
	@kubectl apply -f configs/pytorch-distributed-job.yaml
	@echo -e "$(GREEN)✓ Job submitted: $(JOB_NAME)$(NC)"
	@echo "Use 'make status' to check job status"

.PHONY: delete-job
delete-job: ## Delete PyTorch training job
	@echo -e "$(YELLOW)Deleting PyTorch training job...$(NC)"
	@kubectl delete pytorchjob $(JOB_NAME) || echo "Job not found"
	@echo -e "$(GREEN)✓ Job deleted$(NC)"

.PHONY: status
status: ## Show job and pod status
	@echo -e "$(BLUE)Job Status:$(NC)"
	@echo "==========="
	@kubectl get pytorchjob $(JOB_NAME) -o wide || echo "Job not found"
	@echo ""
	@echo "Pods:"
	@kubectl get pods -l training.kubeflow.org/job-name=$(JOB_NAME) -o wide || echo "No pods found"
	@echo ""
	@echo "Recent Events:"
	@kubectl get events --field-selector involvedObject.name=$(JOB_NAME) --sort-by='.lastTimestamp' | tail -5 || echo "No events found"

.PHONY: logs
logs: ## Show logs from master pod
	@echo -e "$(BLUE)Master Pod Logs:$(NC)"
	@kubectl logs -l training.kubeflow.org/job-name=$(JOB_NAME),training.kubeflow.org/replica-type=master -f --tail=100

# ==============================================================================
# WORKFLOWS
# ==============================================================================

.PHONY: run-e2e-workflow
run-e2e-workflow: ## Run complete end-to-end workflow (training + inference)
	@echo -e "$(BLUE)Running complete end-to-end workflow...$(NC)"
	@chmod +x examples/01-complete-workflow/run-complete-workflow.sh
	@cd examples/01-complete-workflow && ./run-complete-workflow.sh all
	@echo -e "$(GREEN)✓ End-to-end workflow completed!$(NC)"

# ==============================================================================
# DEBUGGING
# ==============================================================================

.PHONY: debug
debug: ## Show debugging information
	@echo -e "$(BLUE)Debugging Information:$(NC)"
	@echo "====================="
	@echo "Cluster Context:"
	@kubectl config current-context
	@echo ""
	@echo "Cluster Info:"
	@kubectl cluster-info
	@echo ""
	@echo "Nodes:"
	@kubectl get nodes
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

# ==============================================================================
# CLEANUP
# ==============================================================================

.PHONY: cleanup
cleanup: ## Clean up all resources (jobs, configmaps)
	@echo -e "$(YELLOW)Cleaning up resources...$(NC)"
	@kubectl delete pytorchjob $(JOB_NAME) || echo "Job not found"
	@kubectl delete configmap pytorch-training-script || echo "ConfigMap not found"
	@echo -e "$(GREEN)✓ Resources cleaned up$(NC)"

.PHONY: cleanup-cluster
cleanup-cluster: cleanup ## Clean up everything including Kind cluster
	@echo -e "$(YELLOW)Deleting Kind cluster...$(NC)"
	@kind delete cluster --name $(CLUSTER_NAME) || echo "Cluster not found"
	@echo -e "$(GREEN)✓ Complete cleanup done$(NC)"

# ==============================================================================
# CONVENIENCE TARGETS
# ==============================================================================

.PHONY: restart
restart: delete-job submit-job ## Restart PyTorch training job

.PHONY: quick-start
quick-start: setup submit-job status ## Quick start: setup and submit job
	@echo -e "$(GREEN)✓ Quick start completed! Use 'make logs' to view progress$(NC)"

# Make setup.sh executable
setup.sh:
	@chmod +x setup.sh

# Default target when no argument is given
.DEFAULT_GOAL := help 