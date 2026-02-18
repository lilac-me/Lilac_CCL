import torch
import torch.distributed as dist


def all_gather_test():
    dist.init_process_group(backend='gloo')
    world_size = dist.get_world_size() # --nproc_per_node=4 means world_size=4
    rank = dist.get_rank() # rank is the unique identifier of each process, ranging from 0 to world_size-1

    # Create a tensor with the rank of the process
    send_tensor = torch.tensor(
        [rank * world_size + i for i in range(world_size)], dtype=torch.int32
    )
    recv_tensor_list = [
        torch.empty_like(send_tensor) for _ in range(world_size)
    ]
    recv_tensor = torch.empty(world_size * send_tensor.numel(), dtype=send_tensor.dtype)

    print(f"Rank {rank} send tensor:\n {send_tensor} \n")

    # Perform all-gather communication
    # all_gather gathers tensors from all processes and concatenates them into a list of tensors
    dist.all_gather(recv_tensor_list, send_tensor)
    # all_gather_into_tensor gathers tensors from all processes and concatenates them into a single tensor
    dist.all_gather_into_tensor(recv_tensor, send_tensor) 

    print(f"all_gather result (list of tensors):\n Rank {rank} received tensor:\n {recv_tensor_list} \n")
    print(f"all_gather_into_tensor result (single concatenated tensor):\n Rank {rank} received tensor:\n {recv_tensor} \n")
    dist.destroy_process_group()


if __name__ == "__main__":
    all_gather_test()

# case: 4 processes, each sending a tensor of shape [4]
# Rank 0 send tensor:
#  tensor([0, 1, 2, 3], dtype=torch.int32)
# Rank 1 send tensor:
#  tensor([4, 5, 6, 7], dtype=torch.int32)
# Rank 2 send tensor:
#  tensor([8, 9, 10, 11], dtype=torch.int32
# Rank 3 send tensor:
#  tensor([12, 13, 14, 15], dtype=torch.int32

# all_gather result (list of tensors):
# Rank 0, 1, 2, 3 received tensor:
#  [tensor([0, 1, 2, 3], dtype=torch.int32),
#   tensor([4, 5, 6, 7], dtype=torch.int32),
#   tensor([8, 9, 10, 11], dtype=torch.int32),
#   tensor([12, 13, 14, 15], dtype=torch.int32)]

# all_gather_into_tensor result (single concatenated tensor):
# Rank 0, 1, 2, 3 received tensor:
#  tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int32)
