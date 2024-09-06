from os import environ

from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.multiprocessing import spawn
from torch.distributed import destroy_process_group, init_process_group
from torch.cuda import device_count, set_device

class DistributedDataLoader:
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.device_count = device_count()

        assert (
            self.batch_size % self.device_count == 0
        ), f"(batch size) % (device count) != 0"
        self.batch_size //= self.device_count

        assert (
            self.num_workers % self.device_count == 0
        ), f"(num workers) % (device count) != 0"
        self.num_workers //= self.device_count

        self.sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        self.loader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.sampler,
            shuffle=False,
        )

    def __iter__(self):
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)


class DistributedParallel(Module):
    def __init__(
        self,
        module: Module,
        device: int,
        find_unused_parameters: bool = True,
    ):
        super().__init__()
        set_device(device)
        module = module.to(device)

        self.distributed = DistributedDataParallel(
            module=module,
            device_ids=[device],
            find_unused_parameters=find_unused_parameters,
        )

    def forward(self, *args, **kwargs):
        return self.distributed(*args, **kwargs)


class DistributedTrainer:
    def __init__(self, func, addr="localhost", port="8888", backend="nccl"):
        environ["MASTER_ADDR"] = addr
        environ["MASTER_PORT"] = port
        self.world_size = device_count()
        self.backend = backend
        self.func = func

    def worker(self, rank, ngpus_per_node):
        init_process_group(
            backend=self.backend,
            init_method="env://",
            world_size=ngpus_per_node,
            rank=rank,
        )

        result = self.func(rank)
        destroy_process_group()
        return result

    def __call__(self):
        spawn(
            self.worker,
            nprocs=self.world_size,
            args=(self.world_size,),
        )
