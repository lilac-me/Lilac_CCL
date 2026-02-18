import torch
import torch.distributed as dist


def all_reduce_test():
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    send_tensor = torch.tensor(
        [rank * world_size + i for i in range(world_size)], dtype=torch.int32,
    )
    recv_tensor = send_tensor.clone()

    print(f"Rank {rank} send tensor:\n {send_tensor}\n")

    # Perform all-reduce communication
    dist.all_reduce(recv_tensor, op=dist.ReduceOp.SUM)

    print(f"Rank {rank} received tensor:\n {recv_tensor}\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    all_reduce_test()

# case: 4 processes, each sending a tensor of shape [4]
# Rank 0 send tensor:
#  tensor([0, 1, 2, 3], dtype=torch.int32)
# Rank 1 send tensor:
#  tensor([4, 5, 6, 7], dtype=torch.int32)
# Rank 2 send tensor:
#  tensor([8, 9, 10, 11], dtype=torch.int32
# Rank 3 send tensor:
#  tensor([12, 13, 14, 15], dtype=torch.int32

# all_reduce result (tensors):
# Rank 0, 1, 2, 3 received tensor:
#  tensor([24, 28, 32, 36], dtype=torch.int32)