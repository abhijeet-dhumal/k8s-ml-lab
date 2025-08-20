def train_func():
    import os
    import sys
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DistributedSampler
    from torchvision import datasets, transforms
    import torch.distributed as dist

    # [1] Setup PyTorch DDP: initialize process group.
    # Force GLOO backend for CPU stability and lower memory usage
    backend = "gloo"  # Always use GLOO for CPU training (more stable)
    dist.init_process_group(backend=backend)
    Distributor = torch.nn.parallel.DistributedDataParallel
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    global_rank = dist.get_rank()

    print(
        f"[Rank {global_rank}] Distributed Training initialized: WORLD_SIZE={dist.get_world_size()}, LOCAL_RANK={local_rank}"
    )
    sys.stdout.flush()

    # [2] Define a CNN Model for MNIST.
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
            self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
            self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
            self.fc2 = torch.nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    # [3] Attach model to CPU device and wrap in DDP.
    # Force CPU usage for stability and lower memory consumption
    device = torch.device("cpu")  # Always use CPU for stability
    model = Net().to(device)
    model = Distributor(model, device_ids=None)  # No GPU device IDs
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.5)  # Reduced learning rate

    # [4] Load MNIST dataset using the mounted input volume from kind cluster.
    # The kind cluster mounts ./input to /input in the container
    dataset_path = "/input"
    download=False
    # Check if the mounted volume exists, fallback to local path if not
    if not os.path.exists(dataset_path):
        dataset_path = "./data"
        download=True
        print(f"[Rank {global_rank}] Warning: Mounted volume /input not found, using local path: {dataset_path}")
    
    print(f"[Rank {global_rank}] Loading MNIST dataset from: {dataset_path}")
    
    dataset = datasets.MNIST(
        dataset_path,
        train=True,
        download=download,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=64,  # Reduced from 128 to 64 for lower memory usage
        sampler=DistributedSampler(dataset),
    )

    # [5] Training loop for 2 epochs (reduced from 3 for faster completion).
    for epoch in range(2):  # Reduced from 3 to 2 epochs
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0 and global_rank == 0:
                print(
                    f"[Rank {global_rank}] Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                    f"({100.0 * batch_idx / len(train_loader):.0f}%)]\tloss={loss.item():.4f}"
                )
                sys.stdout.flush()

    dist.barrier()

    # [6] On rank 0: save the model to the mounted output volume.
    if global_rank == 0:
        # Use the mounted output volume from kind cluster
        output_dir = "/output"
        
        # Check if the mounted volume exists, fallback to local path if not
        if not os.path.exists(output_dir):
            output_dir = "./output"
            print(f"[Rank {global_rank}] Warning: Mounted volume /output not found, using local path: {output_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "mnist_cnn.pt")
        
        # Save the underlying model's state_dict (not the DDP wrapper).
        torch.save(model.module.state_dict(), model_path)
        print(f"[Rank {global_rank}] Model saved at {model_path}")
        
        # Also save training metadata
        metadata_path = os.path.join(output_dir, "training_metadata.txt")
        with open(metadata_path, 'w') as f:
            f.write(f"Training completed successfully\n")
            f.write(f"Model saved at: {model_path}\n")
            f.write(f"Training epochs: 2\n")  # Updated to reflect actual epochs
            f.write(f"Batch size: 64\n")     # Updated to reflect actual batch size
            f.write(f"Learning rate: 0.001\n")  # Updated to reflect actual learning rate
            f.write(f"Backend: {backend}\n")
            f.write(f"World size: {dist.get_world_size()}\n")
        
        print(f"[Rank {global_rank}] Training metadata saved at {metadata_path}")

    # Clean up the distributed process group.
    dist.destroy_process_group()