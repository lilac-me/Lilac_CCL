import torch
import torch.distributed as dist


def reduce_scatter_test():
    """
    Two stages, that is
        Reduce:
            position 0: 0 + 4 + 8 + 12 = 24
            position 1: 1 + 5 + 9 + 13 = 28
            position 2: 2 + 6 + 10 + 14 = 32
            position 3: 3 + 7 + 11 + 15 = 36
    So the intermediate result is all of ranks have [24, 28, 32, 36]
        Scater:
        rank 0 get position 0 is 24
        rank 1 get position 1 is 28
        rank 2 get position 2 is 32
        rank 3 get position 3 is 36
    Note that the data (numel) will be splited as N (world_size) partitions
    """
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()  # --nproc_per_node=4 means world_size=4
    rank = dist.get_rank()  # rank is the unique identifier of each process, ranging from 0 to world_size-1

    # Create a tensor with the rank of the process
    send_tensor = torch.tensor(
        [rank * world_size + i for i in range(world_size)], dtype=torch.int32
    )
    recv_tensor = torch.empty(send_tensor.numel() // world_size, dtype=send_tensor.dtype)

    print(f"Rank {rank} send tensor:\n {send_tensor} \n")

    # Perform reduce-scatter communication
    dist.reduce_scatter_tensor(recv_tensor, send_tensor)

    print(f"Rank {rank} received tensor:\n {recv_tensor} \n")


if __name__ == "__main__":
    reduce_scatter_test()

# case: 4 processes, each sending a tensor of shape [4]
# Rank 0 send tensor:
#  tensor([0, 1, 2, 3], dtype=torch.int32)
# Rank 1 send tensor:
#  tensor([4, 5, 6, 7], dtype=torch.int32)
# Rank 2 send tensor:
#  tensor([8, 9, 10, 11], dtype=torch.int32
# Rank 3 send tensor:
#  tensor([12, 13, 14, 15], dtype=torch.int32

# reduce_scatter_tensor result (tensor):
# Rank 0 received tensor:
#  tensor([24], dtype=torch.int32) 

# Rank 1 received tensor:
#  tensor([28], dtype=torch.int32) 

# Rank 2 received tensor:
#  tensor([32], dtype=torch.int32) 

# Rank 3 received tensor:
#  tensor([36], dtype=torch.int32) 
