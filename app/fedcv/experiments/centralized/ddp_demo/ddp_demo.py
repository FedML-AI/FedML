import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = nn.Linear(10, 10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))


def init_ddp():
    # use InfiniBand
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_SOCKET_IFNAME'] = 'ib0'

    # This the global rank: 0, 1, 2, ..., 15
    global_rank = int(os.environ['RANK'])
    print("int(os.environ['RANK']) = %d" % global_rank)

    # This the globak world_size
    world_size = int(os.environ['WORLD_SIZE'])
    print("world_size = %d" % world_size)

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    local_rank = args.local_rank
    print(f"Running basic DDP example on local rank {local_rank}.")
    return local_rank, global_rank


def get_ddp_model(model, local_rank):
    return DDP(model, device_ids=[local_rank], output_device=local_rank)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DDP Demo")
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    print(args)

    # DDP
    local_rank, global_rank = init_ddp()

    # GPU
    device = torch.device("cuda:" + str(local_rank))

    # Model
    model = ToyModel().to(device)
    model = get_ddp_model(model, local_rank)
    if global_rank == 0:
        print(model)

    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for i in range(100000):
        # 1. forward propagation
        optimizer.zero_grad()
        outputs = model(torch.randn(20, 10))

        # 2. compute loss
        labels = torch.randn(20, 5).to(device)
        loss = loss_fn(outputs, labels)
        print("rank=%d, loss=%f" % (local_rank, loss))

        # 3. backward propagation
        loss.backward()

        # 4. update weight
        optimizer.step()

    dist.destroy_process_group()
