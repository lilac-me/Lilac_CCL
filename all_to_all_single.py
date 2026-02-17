import torch
import torch.distributed as dist

def all_to_all_single_test():
    dist.init_process_group(backend='gloo')
    world_size = dist.get_world_size() # --nproc_per_node=4 means world_size=4
    rank = dist.get_rank() # rank is the unique identifier of each process, ranging from 0 to world_size-1

    # Create a tensor with the rank of the process
    send_tensor = torch.tensor(
        [rank * world_size + i for i in range(world_size)], dtype=torch.int32
    ) # Each process creates a tensor of shape [world_size], with values unique to the process
    recv_tensor = torch.empty_like(send_tensor)

    print(f"Rank {rank} send tensor:\n {send_tensor} \n")

    # Perform all-to-all communication
    dist.all_to_all_single(recv_tensor, send_tensor)

    print(f"Rank {rank} received tensor:\n {recv_tensor} \n")
    dist.destroy_process_group()

if __name__ == "__main__":
    all_to_all_single_test()

# case: 4 processes, each sending a tensor of shape [4]
# Rank 0 send tensor:
#  tensor([0, 1, 2, 3], dtype=torch.int32)
# Rank 1 send tensor:
#  tensor([4, 5, 6, 7], dtype=torch.int32)
# Rank 2 send tensor:
#  tensor([8, 9, 10, 11], dtype=torch.int32)
# Rank 3 send tensor:
#  tensor([12, 13, 14, 15], dtype=torch.int32

# Rank 0 received tensor:
#  tensor([0, 4, 8, 12], dtype=torch.int32)
# Rank 1 received tensor:
#  tensor([1, 5, 9, 13], dtype=torch.int32)
# Rank 2 received tensor:
#  tensor([2, 6, 10, 14], dtype=torch.int32)
# Rank 3 received tensor:
#  tensor([3, 7, 11, 15], dtype=torch.int32)
