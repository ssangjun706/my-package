## Installation
To install the package directly from the GitHub repository, use the following command:

```bash
pip install git+https://github.com/ssangjun706/my-package.git
```

## Upgrade
To update the package to the latest version:

```bash
pip install --upgrade git+https://github.com/ssangjun706/my-package.git
```

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
---
