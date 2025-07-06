#!/usr/bin/env python3

import torch
import torch.distributed as dist
import os
import time
import sys
import datetime

def setup():
    """Simple setup function"""
    try:
        print("Starting simple distributed test...")
        
        # Wait for both pods to be ready
        print("Waiting for pods to be ready...")
        time.sleep(5)
        
        # Initialize with gloo backend (CPU-friendly)
        dist.init_process_group(
            backend='gloo',
            timeout=datetime.timedelta(seconds=30)
        )
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        print(f"SUCCESS: Rank {rank}, World size {world_size}")
        
        # Test communication
        test_tensor = torch.tensor([rank], dtype=torch.float32)
        dist.all_reduce(test_tensor)
        print(f"Rank {rank}: Communication test PASSED - sum: {test_tensor.item()}")
        
        return rank, world_size
        
    except Exception as e:
        print(f"ERROR in setup: {e}")
        sys.exit(1)

def main():
    """Main test function"""
    try:
        rank, world_size = setup()
        
        print(f"Rank {rank}: Starting simple training test...")
        
        # Simple computation
        for i in range(3):
            print(f"Rank {rank}: Iteration {i}")
            time.sleep(1)
        
        # Save a simple file (only rank 0)
        if rank == 0:
            os.makedirs('/output', exist_ok=True)
            with open('/output/test_success.txt', 'w') as f:
                f.write(f'Test completed successfully with {world_size} workers\n')
            print("Rank 0: Test file saved")
        
        # Clean up
        dist.destroy_process_group()
        print(f"Rank {rank}: Test completed successfully!")
        
        # Exit with success code
        sys.exit(0)
        
    except Exception as e:
        print(f"ERROR in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 