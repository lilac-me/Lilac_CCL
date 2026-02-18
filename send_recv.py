import torch 
import torch.distributed as dist


def send_recv_test():
    dist.init_process_group(backend="gloo")
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    send_tensor = torch.tensor(
        [rank * world_size + i for i in range(world_size)], dtype=torch.int32
    )
    recv_tensor = torch.empty_like(send_tensor)

    print(f"Rank {rank} send tensor: {send_tensor}\n")

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    send_req = dist.isend(send_tensor, dst=next_rank)
    recv_req = dist.irecv(recv_tensor, src=prev_rank)

    send_req.wait()
    recv_req.wait()

    print(f"Rank {rank} finished, received tensor: {recv_tensor}")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    send_recv_test()
