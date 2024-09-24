## Installation
To install the package directly from the GitHub repository, use the following command:

```bash
pip install git+https://github.com/ssangjun706/my_package.git
```

## Upgrade
To update the package to the latest version:

```bash
pip install --upgrade git+https://github.com/ssangjun706/my_package.git
```

## Module: `parallel`
It offers simplified PyTorch distributed training. It focuses on making distributed workflows more accessible and removing unnecessary boilerplate. As primarily intended for personal use, it does not provide any additional features beyond the core functionality.

### Features

- **`DistributedDataLoader`**: A data loader that manages data distribution across multiple processes.
- **`DistributedParallel`**: A wrapper for PyTorch models to enable easy parallelization across devices.
- **`DistributedTrainer`**: A training framework that supports both return and yield values during distributed training.

### Usage Example

```python
from parallel import DistributedDataLoader, DistributedParallel, DistributedTrainer

# Define the training function
def train(rank):
    """
    Training function for distributed training.
    
    Args:
        rank (int): The rank of the current process in distributed training.
    """
    # Initialize your model
    model = MyModel()       

    # Wrap the model with DistributedParallel for multi-device training
    model = DistributedParallel(model, device=rank)

    # Load your dataset
    dataset = MyDataset()   
    
    # Create a distributed data loader to handle batching and distribution
    loader = DistributedDataLoader(dataset, batch_size)

    # Iterate over the data loader for training
    for batch in tqdm(loader):
        # Training logic goes here...

        # Optionally yield values (e.g., loss, accuracy)
        yield values
    
    # Optionally return final result (cannot be used with yield)
    return result

if __name__ == "__main__":
    """
    Main entry point for the script. Initializes and starts distributed training.
    """
    # Initialize the DistributedTrainer with the training function
    trainer = DistributedTrainer(train)

    # Option 1: Use return values from the trainer
    result = trainer()
    
    # Option 2: Use yield values from the trainer
    for values in trainer:
        # Handle the yielded values here
        pass
```

### Update Notes
- **Separation of Return and Yield Values**: The `DistributedTrainer` class supports both return and yield methods for handling values during the training process. This allows you to either collect final results or continuously monitor metrics during training.
---
