"""
Integration guide: Ascend 512-byte alignment in verl's FSDPSFTTrainer.

在 verl trainer 代码中只需要改两处。
"""

# =============================================================================
# 改法一（推荐）：在 trainer 初始化后、训练前 apply hook
# =============================================================================

# 在 main() 函数里，trainer 初始化之后加一行：

"""
    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    
    # >>> 新增：Ascend 512 对齐 <<<
    from ascend_alignment import apply_ascend_alignment_to_trainer, check_fsdp_alignment
    apply_ascend_alignment_to_trainer(trainer)
    check_fsdp_alignment(trainer.fsdp_model)  # 可选：打印诊断信息
    
    trainer.fit()
"""


# =============================================================================
# 改法二：直接改 _build_model_optimizer 方法
# =============================================================================

# 在 FSDPSFTTrainer._build_model_optimizer 方法中，FSDP 初始化之后加 hook：

"""
    def _build_model_optimizer(self):
        ...
        
        self.fsdp_model = FSDP(
            module=self.model,
            auto_wrap_policy=auto_wrap_policy,
            param_init_fn=init_fn,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            device_mesh=self.device_mesh,
            sync_module_states=True,
            device_id=torch.cuda.current_device(),
            cpu_offload=cpu_offload,
            use_orig_params=False,
        )
        
        # >>> 新增：Ascend 512-byte alignment for HCCL <<<
        from ascend_alignment import register_ascend_alignment_hooks, check_fsdp_alignment
        register_ascend_alignment_hooks(self.fsdp_model)
        check_fsdp_alignment(self.fsdp_model, verbose=True)
        
        log_gpu_memory_usage("After FSDP wrapping", logger=logger)
        ...
"""


# =============================================================================
# 改法三（最彻底）：Override MixedPrecision 的 reduce_dtype buffer 大小
# =============================================================================

# 如果上面的 hook 方式不够（比如 PyTorch 版本太老不支持 comm hook），
# 可以直接 monkey-patch torch.distributed 的 allgather / reduce_scatter：

"""
import torch.distributed as dist
from ascend_alignment import pad_tensor_for_comm, unpad_tensor_from_comm, ASCEND_ALIGNMENT_BYTES

_orig_all_gather_into_tensor = dist.all_gather_into_tensor
_orig_reduce_scatter_tensor = dist.reduce_scatter_tensor

def _aligned_all_gather_into_tensor(output, input_tensor, group=None, async_op=False):
    padded_input, pad_size = pad_tensor_for_comm(input_tensor, ASCEND_ALIGNMENT_BYTES)
    if pad_size > 0:
        world_size = dist.get_world_size(group)
        padded_output = torch.empty(
            padded_input.numel() * world_size,
            dtype=input_tensor.dtype,
            device=input_tensor.device,
        )
        result = _orig_all_gather_into_tensor(padded_output, padded_input, group=group, async_op=async_op)
        # Unpad each chunk
        chunk_size = input_tensor.numel()
        padded_chunk_size = padded_input.numel()
        for i in range(world_size):
            src_start = i * padded_chunk_size
            dst_start = i * chunk_size
            output[dst_start:dst_start + chunk_size].copy_(
                padded_output[src_start:src_start + chunk_size]
            )
        return result
    else:
        return _orig_all_gather_into_tensor(output, input_tensor, group=group, async_op=async_op)

def _aligned_reduce_scatter_tensor(output, input_tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
    world_size = dist.get_world_size(group)
    chunk_size = input_tensor.numel() // world_size
    from ascend_alignment import align_numel
    padded_chunk_size = align_numel(chunk_size, input_tensor.dtype)
    pad_per_chunk = padded_chunk_size - chunk_size
    
    if pad_per_chunk > 0:
        chunks = input_tensor.view(world_size, chunk_size)
        padding = torch.zeros(world_size, pad_per_chunk, dtype=input_tensor.dtype, device=input_tensor.device)
        padded_input = torch.cat([chunks, padding], dim=1).contiguous().view(-1)
        padded_output = torch.empty(padded_chunk_size, dtype=input_tensor.dtype, device=input_tensor.device)
        result = _orig_reduce_scatter_tensor(padded_output, padded_input, op=op, group=group, async_op=async_op)
        output.copy_(padded_output[:chunk_size])
        return result
    else:
        return _orig_reduce_scatter_tensor(output, input_tensor, op=op, group=group, async_op=async_op)

# Apply monkey-patch
dist.all_gather_into_tensor = _aligned_all_gather_into_tensor
dist.reduce_scatter_tensor = _aligned_reduce_scatter_tensor
"""