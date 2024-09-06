# My package

## Installation

```bash
pip install git+https://github.com/ssangjun706/my_package.git
```

## Module `parallel`

### Classes

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
    # Disable progress bar for all devices except the main one (rank 0)
    for X, y in tqdm(loader, disable=device!=0):
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
