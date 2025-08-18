import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import os
import time

# Configure logger.
log_formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%SZ"
)
logger = logging.getLogger(__file__)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.setLevel(logging.INFO)
    
def ddp_setup(backend="nccl"):
    """Setup for Distributed Data Parallel with specified backend."""
    try:
        # If CUDA is not available, use CPU as the fallback
        if torch.cuda.is_available() and backend=="nccl":
            # Check GPU availability
            num_devices = torch.cuda.device_count()
            device = int(os.environ.get("LOCAL_RANK", 0))  # Default to device 0
            if device >= num_devices:
                logger.warning(f"Warning: Invalid device ordinal {device}. Defaulting to device 0.")
                device = 0
            torch.cuda.set_device(device)
            logger.info(f"Using GPU device {device}: {torch.cuda.get_device_name(device)}")
        else:
            # If no GPU is available, use Gloo backend (for CPU-only environments)
            logger.info("No GPU available, falling back to CPU with GLOO backend.")
            backend="gloo"
        
        dist.init_process_group(backend=backend)
        logger.info(f"Distributed training initialized with {backend} backend")
        
    except Exception as e:
        logger.error(f"Failed to initialize distributed training: {e}")
        raise

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        backend: str,
        lightweight_checkpoints: bool = False,
    ) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))  # Ensure fallback if LOCAL_RANK isn't set
        self.global_rank = int(os.environ.get("RANK", 0))
        
        # Validate device rank
        if self.local_rank < 0:
            self.local_rank = 0
            logger.warning(f"Invalid LOCAL_RANK, defaulting to 0")

        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.backend = backend
        self.lightweight_checkpoints = lightweight_checkpoints
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        if os.path.exists(snapshot_path):
            logger.info("Loading snapshot")
            self._load_snapshot(snapshot_path)

        # Move model to the appropriate device (GPU/CPU)
        if torch.cuda.is_available() and self.backend=="nccl":
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.model = DDP(self.model.to(self.device), device_ids=[self.local_rank])
            logger.info(f"Model moved to GPU {self.local_rank}: {torch.cuda.get_device_name(self.local_rank)}")
        else:
            self.device = torch.device('cpu')
            self.model = DDP(self.model.to(self.device))
            logger.info("Model moved to CPU")
        
        logger.info(f"Using device: {self.device}")

    def _load_snapshot(self, snapshot_path):
        """Load training snapshot with error handling."""
        try:
            snapshot = torch.load(snapshot_path, map_location=self.device)
            self.model.load_state_dict(snapshot["MODEL_STATE"])
            self.epochs_run = snapshot["EPOCHS_RUN"]
            
            # Load optimizer state if available
            if "OPTIMIZER_STATE" in snapshot:
                self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
            
            # Load training history if available
            if "TRAIN_HISTORY" in snapshot:
                self.train_losses = snapshot["TRAIN_HISTORY"].get("losses", [])
                self.train_accuracies = snapshot["TRAIN_HISTORY"].get("accuracies", [])
                self.val_losses = snapshot["TRAIN_HISTORY"].get("val_losses", [])
                self.val_accuracies = snapshot["TRAIN_HISTORY"].get("val_accuracies", [])
            
            logger.info(f"Snapshot loaded successfully from epoch {self.epochs_run}")
            
        except Exception as e:
            logger.error(f"Failed to load snapshot: {e}")
            logger.info("Starting training from scratch")

    def _run_batch(self, source, targets):
        """Run a single training batch with metrics tracking."""
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        
        # Calculate accuracy
        pred = output.argmax(dim=1)
        accuracy = (pred == targets).float().mean()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), accuracy.item()

    def _run_epoch(self, epoch, backend):
        """Run a single training epoch with comprehensive logging."""
        b_sz = len(next(iter(self.train_data))[0])
        if torch.cuda.is_available() and backend=="nccl":
            logger.info(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        else:
            logger.info(f"[CPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        
        if isinstance(self.train_data.sampler, DistributedSampler):
            self.train_data.sampler.set_epoch(epoch)
        
        self.model.train()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (source, targets) in enumerate(self.train_data):
            source = source.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            loss, accuracy = self._run_batch(source, targets)
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1
            
            # Progress logging every 50 batches
            if batch_idx % 50 == 0:
                avg_loss = total_loss / num_batches
                avg_accuracy = total_accuracy / num_batches
                logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_data)} | "
                          f"Loss: {avg_loss:.4f} | Acc: {avg_accuracy*100:.1f}%")
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Store training metrics
        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)
        
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s | "
                   f"Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_accuracy*100:.1f}%")
        
        return avg_loss, avg_accuracy

    def validate(self, epoch):
        """Run validation on the validation set."""
        if self.val_data is None:
            return 0.0, 0.0
        
        self.model.eval()
        total_loss = 0
        total_accuracy = 0
        num_batches = 0
        
        logger.info(f"Starting validation for epoch {epoch}")
        
        with torch.no_grad():
            for source, targets in self.val_data:
                source = source.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                output = self.model(source)
                loss = F.cross_entropy(output, targets)
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                accuracy = (pred == targets).float().mean()
                
                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches
        
        # Store validation metrics
        self.val_losses.append(avg_loss)
        self.val_accuracies.append(avg_accuracy)
        
        logger.info(f"Validation Epoch {epoch} | Loss: {avg_loss:.4f} | Acc: {avg_accuracy*100:.1f}%")
        
        return avg_loss, avg_accuracy

    def _save_snapshot(self, epoch):
        """Save memory-efficient training snapshot."""
        try:
            # Get model state dict (handle both DDP and regular models)
            if hasattr(self.model, 'module'):
                model_state = self.model.module.state_dict()
            else:
                model_state = self.model.state_dict()
            
            # MEMORY OPTIMIZATION: Smart checkpointing based on settings
            if self.lightweight_checkpoints:
                # Lightweight checkpoint (model + optimizer only) - saves memory
                snapshot = {
                    "MODEL_STATE": model_state,
                    "OPTIMIZER_STATE": self.optimizer.state_dict(),
                    "EPOCHS_RUN": epoch
                }
                logger.info(f"Epoch {epoch} | Lightweight snapshot saved (memory efficient)")
            else:
                # Full checkpoint with history (more memory but complete tracking)
                snapshot = {
                    "MODEL_STATE": model_state,
                    "OPTIMIZER_STATE": self.optimizer.state_dict(),
                    "EPOCHS_RUN": epoch,
                    "TRAIN_HISTORY": {
                        "losses": self.train_losses,
                        "accuracies": self.train_accuracies,
                        "val_losses": self.val_losses,
                        "val_accuracies": self.val_accuracies
                    }
                }
                logger.info(f"Epoch {epoch} | Full snapshot saved at {self.snapshot_path}")
            
            torch.save(snapshot, self.snapshot_path)
            
        except Exception as e:
            logger.error(f"Failed to save snapshot: {e}")

    def train(self, max_epochs: int, backend: str):
        """Main training loop with validation."""
        logger.info(f"Starting training for {max_epochs} epochs")
        
        best_val_accuracy = 0
        
        for epoch in range(self.epochs_run, max_epochs):
            # Training phase
            train_loss, train_acc = self._run_epoch(epoch, backend)
            
            # Validation phase
            val_loss, val_acc = self.validate(epoch)
            
            # Save snapshot periodically
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                if self.global_rank == 0:
                    best_model_path = self.snapshot_path.replace('.pt', '_best.pt')
                    torch.save({
                        "MODEL_STATE": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                        "EPOCHS_RUN": epoch,
                        "VAL_ACCURACY": val_acc
                    }, best_model_path)
                    logger.info(f"New best model saved with validation accuracy: {val_acc*100:.1f}%")
        
        logger.info(f"Training completed! Best validation accuracy: {best_val_accuracy*100:.1f}%")
        
        # Save final model and checkpoint (restore previous structure)
        if self.global_rank == 0:
            # Extract output directory from snapshot path
            output_dir = os.path.dirname(self.snapshot_path)
            if not output_dir:
                output_dir = "."
            
            # Save final model (like previous version)
            final_model_path = os.path.join(output_dir, 'mnist_model.pth')
            if hasattr(self.model, 'module'):
                torch.save(self.model.module.state_dict(), final_model_path)
            else:
                torch.save(self.model.state_dict(), final_model_path)
            
            # Create checkpoints directory and save final checkpoint (like previous version)
            checkpoint_dir = os.path.join(output_dir, 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            final_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
            torch.save({
                "MODEL_STATE": self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
                "EPOCHS_RUN": max_epochs,
                "FINAL_ACCURACY": best_val_accuracy
            }, final_checkpoint_path)
            
            logger.info(f"Final model saved to: {final_model_path}")
            logger.info(f"Final checkpoint saved to: {final_checkpoint_path}")


def load_train_objs(dataset_path: str, lr: float):
    """Load dataset, model, and optimizer with FIXED train parameter."""
    try:
        # FIXED: train=True for training data, train=False for validation
        train_set = MNIST(dataset_path, train=True, download=True, 
                         transform=transforms.Compose([
                             transforms.ToTensor(),
                             transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
                         ]))
        
        val_set = MNIST(dataset_path, train=False, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ]))
        
        model = Net()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        
        logger.info(f"Dataset loaded: {len(train_set)} training, {len(val_set)} validation samples")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return train_set, val_set, model, optimizer
        
    except Exception as e:
        logger.error(f"Failed to load training objects: {e}")
        raise


def prepare_dataloader(dataset: Dataset, batch_size: int, useGpu: bool, is_train: bool = True):
    """Prepare DataLoader with DistributedSampler."""
    try:
        sampler = DistributedSampler(dataset, shuffle=is_train)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=useGpu,
            shuffle=False,  # Sampler handles shuffling
            sampler=sampler,
            num_workers=2 if useGpu else 1,  # More workers for GPU
            drop_last=is_train  # Drop incomplete batches for training
        )
    except Exception as e:
        logger.error(f"Failed to prepare dataloader: {e}")
        raise


def main(epochs: int, save_every: int, batch_size: int, lr: float, dataset_path: str, snapshot_path: str, backend: str, lightweight_checkpoints: bool = False):
    """Main training function with comprehensive error handling."""
    try:
        # Setup distributed training
        ddp_setup(backend)
        
        # Load training objects
        train_dataset, val_dataset, model, optimizer = load_train_objs(dataset_path, lr)
        
        # Prepare data loaders
        train_loader = prepare_dataloader(train_dataset, batch_size, torch.cuda.is_available() and backend=="nccl", is_train=True)
        val_loader = prepare_dataloader(val_dataset, batch_size, torch.cuda.is_available() and backend=="nccl", is_train=False)
        
        # Create trainer
        trainer = Trainer(model, train_loader, val_loader, optimizer, save_every, snapshot_path, backend, args.lightweight_checkpoints)
        
        # Start training
        trainer.train(epochs, backend)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        # Cleanup distributed training
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Distributed training cleaned up")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed MNIST Training")
    parser.add_argument('--epochs', type=int, required=True, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, required=True, help='How often to save a snapshot')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size on each device (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--dataset_path', type=str, default="../input", help='Path to MNIST datasets (default: ../input)')
    parser.add_argument('--snapshot_path', type=str, default="snapshot_mnist.pt", help='Path to save snapshots (default: snapshot_mnist.pt)')
    parser.add_argument('--backend', type=str, choices=['gloo', 'nccl'], default='nccl', help='Distributed backend type (default: nccl)')
    parser.add_argument('--lightweight-checkpoints', action='store_true', help='Use lightweight checkpoints to save memory (default: false)')
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        save_every=args.save_every,
        batch_size=args.batch_size,
        lr=args.lr,
        dataset_path=args.dataset_path,
        snapshot_path=args.snapshot_path,
        backend=args.backend
    )