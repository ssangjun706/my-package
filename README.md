# My package

## Installation

```bash
pip install git+https://github.com/ssangjun706/my_package.git
```

## Module `parallel`
This is designed to simplify distributed training workflows in PyTorch. It was created primarily for personal use to avoid the repetitive boilerplate code associated with DistributedDataParallel. This package has no additional features beyond the basic functionality.

### Features

- `DistributedDataLoader`
- `DistributedParallel`
- `DistributedTrainer`

### Usage Example

```python
from parallel import DistributedDataLoader, DistributedParallel, DistributedTrainer

# Trainer function
def train(rank):
    """
    Training function for distributed training.
    
    Args:
        rank (int): The rank of the current process in distributed training.
    """
    # Initialize the model
    model = MyModel()       

    # Wrap the model with DistributedParallel for distributed training
    model = DistributedParallel(model, device=rank)

    # Load the dataset
    dataset = MyDataset()   
    
    # Create a distributed data loader to handle batching and distribution
    loader = DistributedDataLoader(dataset, batch_size)

    # Iterate over the data loader for training
    for batch in tqdm(loader):
        # Training logic here...
        pass

if __name__ == "__main__":
    """
    Main entry point for the script. Initializes and starts the distributed training.
    """

    # Create a DistributedTrainer instance with the training function
    trainer = DistributedTrainer(train)
    # Start the distributed training process
    trainer()

```

### Integrating with Logging Libraries
To avoid logging duplication, make sure to initialize and log only in the process with rank 0:
```python
import wandb

def train(rank):
    # Initialize wandb only for rank 0
    if rank == 0:
        wandb.init(project='my_package')
    
    # Your training code here...
    
    # Log metrics only for rank 0
    if rank == 0:
        wandb.log({"loss": loss})
```
