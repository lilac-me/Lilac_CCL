"""
Ascend 910B 512-byte address alignment for FSDP AllGather / ReduceScatter.

Ascend HCCL requires communication buffers to be 512-byte aligned for optimal performance.
For FSDP, this means each FlatParameter's numel should be a multiple of (512 / element_size).

This module provides:
1. `get_aligned_numel()` - Calculate aligned numel for a given dtype.
2. `AscendAlignedFSDP` - A thin FSDP wrapper that pads flat params to alignment.
3. `pad_tensor_for_comm()` / `unpad_tensor_from_comm()` - Manual pad/unpad utilities.
4. `align_fsdp_params()` - Post-init hook to verify/fix alignment on existing FSDP model.
5. Integration guide for verl's FSDPSFTTrainer.

Usage in verl trainer:
    Replace the FSDP(...) call with AscendAlignedFSDP(...), or apply
    `register_ascend_alignment_hooks()` after FSDP initialization.
"""

import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor, nn


# =============================================================================
# Constants
# =============================================================================

ASCEND_ALIGNMENT_BYTES = 512


# =============================================================================
# Core alignment utilities
# =============================================================================

def get_alignment_elements(dtype: torch.dtype, alignment_bytes: int = ASCEND_ALIGNMENT_BYTES) -> int:
    """
    Calculate the number of elements needed for alignment.
    
    For bf16 (2 bytes): 512 / 2 = 256 elements
    For fp32 (4 bytes): 512 / 4 = 128 elements
    For fp16 (2 bytes): 512 / 2 = 256 elements
    """
    element_size = torch.tensor([], dtype=dtype).element_size()
    return alignment_bytes // element_size


def align_numel(numel: int, dtype: torch.dtype, alignment_bytes: int = ASCEND_ALIGNMENT_BYTES) -> int:
    """Round up numel to the nearest multiple of alignment elements."""
    align_elems = get_alignment_elements(dtype, alignment_bytes)
    return math.ceil(numel / align_elems) * align_elems


def is_aligned(tensor: Tensor, alignment_bytes: int = ASCEND_ALIGNMENT_BYTES) -> bool:
    """Check if a tensor's data pointer and numel are both aligned."""
    ptr_aligned = (tensor.data_ptr() % alignment_bytes) == 0
    elem_aligned = (tensor.numel() * tensor.element_size()) % alignment_bytes == 0
    return ptr_aligned and elem_aligned


# =============================================================================
# Pad / Unpad for communication
# =============================================================================

def pad_tensor_for_comm(
    tensor: Tensor,
    alignment_bytes: int = ASCEND_ALIGNMENT_BYTES,
) -> Tuple[Tensor, int]:
    """
    Pad a 1D tensor (or flattened view) to 512-byte alignment.
    
    Args:
        tensor: Input tensor, typically a flattened parameter or comm buffer.
        alignment_bytes: Target alignment in bytes.
    
    Returns:
        (padded_tensor, pad_size): The padded tensor and number of padding elements added.
    """
    align_elems = get_alignment_elements(tensor.dtype, alignment_bytes)
    remainder = tensor.numel() % align_elems
    if remainder == 0:
        return tensor, 0
    
    pad_size = align_elems - remainder
    padded = torch.zeros(
        tensor.numel() + pad_size,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:tensor.numel()].copy_(tensor.view(-1))
    return padded, pad_size


def unpad_tensor_from_comm(
    tensor: Tensor,
    original_numel: int,
) -> Tensor:
    """
    Remove padding added by pad_tensor_for_comm.
    
    Args:
        tensor: Padded tensor from communication.
        original_numel: Original number of elements before padding.
    
    Returns:
        Tensor with padding removed.
    """
    return tensor[:original_numel]


# =============================================================================
# Aligned AllGather / ReduceScatter wrappers
# =============================================================================

def aligned_allgather(
    tensor: Tensor,
    group: Optional[dist.ProcessGroup] = None,
    alignment_bytes: int = ASCEND_ALIGNMENT_BYTES,
) -> Tensor:
    """
    AllGather with 512-byte aligned buffers.
    
    1. Pad local tensor to alignment.
    2. AllGather padded tensors.
    3. Unpad to recover original total size.
    
    Args:
        tensor: Local shard tensor (1D).
        group: Process group for communication.
        alignment_bytes: Target alignment.
    
    Returns:
        Gathered tensor with padding removed.
    """
    world_size = dist.get_world_size(group)
    original_numel = tensor.numel()
    
    # Pad local shard
    padded_local, pad_size = pad_tensor_for_comm(tensor, alignment_bytes)
    padded_numel = padded_local.numel()
    
    # Allocate aligned output buffer
    output = torch.empty(
        padded_numel * world_size,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    
    # AllGather
    dist.all_gather_into_tensor(output, padded_local, group=group)
    
    # Unpad: each shard contributed padded_numel elements, but only original_numel are valid
    # Reshape to [world_size, padded_numel], slice valid, then flatten
    chunks = output.view(world_size, padded_numel)
    valid_chunks = chunks[:, :original_numel]
    return valid_chunks.contiguous().view(-1)


def aligned_reduce_scatter(
    tensor: Tensor,
    group: Optional[dist.ProcessGroup] = None,
    op: dist.ReduceOp = dist.ReduceOp.SUM,
    alignment_bytes: int = ASCEND_ALIGNMENT_BYTES,
) -> Tensor:
    """
    ReduceScatter with 512-byte aligned buffers.
    
    1. Pad input tensor so each chunk is aligned.
    2. ReduceScatter on padded buffer.
    3. Unpad the local result.
    
    Args:
        tensor: Full tensor to reduce-scatter (1D), size should be divisible by world_size.
        group: Process group.
        op: Reduction operation.
        alignment_bytes: Target alignment.
    
    Returns:
        Local shard after reduce-scatter, with padding removed.
    """
    world_size = dist.get_world_size(group)
    total_numel = tensor.numel()
    chunk_size = total_numel // world_size
    
    align_elems = get_alignment_elements(tensor.dtype, alignment_bytes)
    padded_chunk_size = math.ceil(chunk_size / align_elems) * align_elems
    pad_per_chunk = padded_chunk_size - chunk_size
    
    if pad_per_chunk == 0:
        # Already aligned, do regular reduce_scatter
        output = torch.empty(chunk_size, dtype=tensor.dtype, device=tensor.device)
        dist.reduce_scatter_tensor(output, tensor, op=op, group=group)
        return output
    
    # Reshape into chunks, pad each chunk, then flatten
    chunks = tensor.view(world_size, chunk_size)
    if pad_per_chunk > 0:
        padding = torch.zeros(
            world_size, pad_per_chunk, dtype=tensor.dtype, device=tensor.device
        )
        padded_chunks = torch.cat([chunks, padding], dim=1)  # [world_size, padded_chunk_size]
    padded_input = padded_chunks.contiguous().view(-1)
    
    # Allocate aligned output
    padded_output = torch.empty(
        padded_chunk_size, dtype=tensor.dtype, device=tensor.device
    )
    
    # ReduceScatter
    dist.reduce_scatter_tensor(padded_output, padded_input, op=op, group=group)
    
    # Unpad
    return padded_output[:chunk_size].contiguous()


# =============================================================================
# FSDP integration: pre/post communication hooks
# =============================================================================

class AscendAlignmentState:
    """Stores alignment metadata for FSDP communication hooks."""
    
    def __init__(self, alignment_bytes: int = ASCEND_ALIGNMENT_BYTES):
        self.alignment_bytes = alignment_bytes
        # Cache for original numels before padding
        self._original_numels = {}


def _ascend_allgather_pre_hook(state: AscendAlignmentState, module, *args):
    """
    FSDP pre-forward hook: pad flat params before allgather.
    
    This hook runs before FSDP's internal allgather. It pads the
    flat parameter shards to 512-byte alignment.
    """
    if not hasattr(module, '_fsdp_wrapped_module'):
        return
    
    for handle in module._handles:
        flat_param = handle.flat_param
        if flat_param is not None and not is_aligned(flat_param, state.alignment_bytes):
            original_numel = flat_param.numel()
            state._original_numels[id(flat_param)] = original_numel
            aligned_numel = align_numel(original_numel, flat_param.dtype, state.alignment_bytes)
            if aligned_numel > original_numel:
                pad_size = aligned_numel - original_numel
                flat_param.data = torch.cat([
                    flat_param.data,
                    torch.zeros(pad_size, dtype=flat_param.dtype, device=flat_param.device)
                ])


def register_ascend_alignment_hooks(
    fsdp_model: nn.Module,
    alignment_bytes: int = ASCEND_ALIGNMENT_BYTES,
) -> AscendAlignmentState:
    """
    Register pre/post communication hooks on an FSDP model for Ascend 512-byte alignment.
    
    This is the simplest integration path - call this after FSDP wrapping:
    
        fsdp_model = FSDP(model, ...)
        alignment_state = register_ascend_alignment_hooks(fsdp_model)
    
    Args:
        fsdp_model: The FSDP-wrapped model.
        alignment_bytes: Alignment requirement in bytes.
    
    Returns:
        AscendAlignmentState for managing alignment metadata.
    """
    state = AscendAlignmentState(alignment_bytes)
    
    # Use FSDP's communication hook API if available (PyTorch >= 2.1)
    try:
        from torch.distributed.fsdp import communication_hook
        # Register custom allgather hook
        # Note: This API varies by PyTorch version
        fsdp_model.register_comm_hook(state, _aligned_comm_hook)
        print("[Ascend] Registered FSDP communication hook for 512-byte alignment")
    except (ImportError, AttributeError):
        # Fallback: register as forward pre-hooks on FSDP submodules
        for name, mod in fsdp_model.named_modules():
            if isinstance(mod, type(fsdp_model)):  # is FSDP instance
                mod.register_forward_pre_hook(
                    lambda m, args, s=state: _ascend_allgather_pre_hook(s, m, args)
                )
        print("[Ascend] Registered forward pre-hooks for 512-byte alignment (fallback)")
    
    return state


def _aligned_comm_hook(state: AscendAlignmentState, bucket):
    """
    FSDP communication hook that pads gradient buckets to 512-byte alignment
    before reduce-scatter.
    """
    tensor = bucket.buffer()
    original_numel = tensor.numel()
    
    padded, pad_size = pad_tensor_for_comm(tensor, state.alignment_bytes)
    
    # Perform allreduce or reduce_scatter on padded tensor
    group = bucket.process_group()
    
    fut = dist.reduce_scatter_tensor(
        padded[:padded.numel() // dist.get_world_size(group)],
        padded,
        group=group,
        async_op=True,
    ).get_future()
    
    def unpad_callback(fut):
        result = fut.value()
        return result[:original_numel // dist.get_world_size(group)]
    
    return fut.then(unpad_callback)


# =============================================================================
# Approach 2: Pad model parameters before FSDP wrapping (recommended for verl)
# =============================================================================

class AlignedLinear(nn.Module):
    """
    A Linear layer whose weight numel is padded to 512-byte alignment.
    
    This is achieved by making the weight's innermost dim a multiple of
    alignment_elements. The extra columns are zeroed and masked out in forward.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        alignment_bytes: int = ASCEND_ALIGNMENT_BYTES,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        align_elems = get_alignment_elements(dtype, alignment_bytes)
        # Pad in_features to alignment
        self.in_features_padded = math.ceil(in_features / align_elems) * align_elems
        
        self.weight = nn.Parameter(torch.zeros(out_features, self.in_features_padded, dtype=dtype))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        # Only use the valid columns of weight
        weight = self.weight[:, :self.in_features]
        return nn.functional.linear(x, weight, self.bias)


def pad_flat_param_numel(
    model: nn.Module,
    alignment_bytes: int = ASCEND_ALIGNMENT_BYTES,
) -> dict:
    """
    Pad each parameter's underlying storage so that numel is aligned.
    Call this BEFORE FSDP wrapping.
    
    This works by:
    1. Flattening each param to 1D
    2. Padding to aligned numel with zeros
    3. Storing original shape for unpadding after communication
    
    NOTE: This changes param shapes. The model's forward must handle this.
    For standard HF models, it's safer to use the hook-based approach instead.
    
    Args:
        model: The model before FSDP wrapping.
        alignment_bytes: Alignment requirement.
    
    Returns:
        Dict mapping param name -> original shape, for later unpadding.
    """
    original_shapes = {}
    
    for name, param in model.named_parameters():
        original_shapes[name] = param.shape
        flat = param.data.view(-1)
        aligned_numel = align_numel(flat.numel(), param.dtype, alignment_bytes)
        
        if aligned_numel > flat.numel():
            pad_size = aligned_numel - flat.numel()
            padded = torch.cat([flat, torch.zeros(pad_size, dtype=param.dtype, device=param.device)])
            param.data = padded
    
    return original_shapes


# =============================================================================
# Approach 3 (Recommended): Custom FSDP extensions for verl trainer
# =============================================================================

def create_aligned_fsdp_buffer_hook():
    """
    Create a pair of hooks (pre-allgather, post-allgather) that handle
    512-byte alignment transparently within FSDP's communication.
    
    Returns pre_hook_fn and post_hook_fn to be registered on FSDP modules.
    """
    _pad_cache = {}
    
    def pre_allgather_hook(module, input):
        """Pad flat params before FSDP's allgather."""
        for handle in getattr(module, '_handles', []):
            fp = handle.flat_param
            if fp is not None:
                numel = fp.numel()
                aligned = align_numel(numel, fp.dtype)
                if aligned != numel:
                    _pad_cache[id(fp)] = numel
                    pad_size = aligned - numel
                    fp.data = torch.cat([
                        fp.data,
                        torch.zeros(pad_size, dtype=fp.dtype, device=fp.device)
                    ])
    
    def post_allgather_hook(module, input, output):
        """Unpad flat params after FSDP's allgather."""
        for handle in getattr(module, '_handles', []):
            fp = handle.flat_param
            if fp is not None and id(fp) in _pad_cache:
                original_numel = _pad_cache.pop(id(fp))
                fp.data = fp.data[:original_numel]
    
    return pre_allgather_hook, post_allgather_hook


# =============================================================================
# Integration function for verl's FSDPSFTTrainer
# =============================================================================

def apply_ascend_alignment_to_trainer(trainer, alignment_bytes: int = ASCEND_ALIGNMENT_BYTES):
    """
    Apply 512-byte alignment to an existing verl FSDPSFTTrainer.
    
    Call this after trainer.__init__() but before trainer.fit():
    
        trainer = FSDPSFTTrainer(config, ...)
        apply_ascend_alignment_to_trainer(trainer)
        trainer.fit()
    
    This patches the trainer's allgather/reducescatter operations to use
    aligned buffers on Ascend 910B.
    """
    fsdp_model = trainer.fsdp_model
    
    # Method 1: Try FSDP communication hooks (cleanest)
    state = AscendAlignmentState(alignment_bytes)
    
    try:
        # PyTorch >= 2.1 with FSDP comm hook support
        fsdp_model.register_comm_hook(state, _aligned_comm_hook)
        if trainer.device_mesh.get_rank() == 0:
            print(f"[Ascend] Applied 512-byte alignment via FSDP comm hook")
        return
    except (AttributeError, RuntimeError):
        pass
    
    # Method 2: Fallback to forward hooks
    pre_hook, post_hook = create_aligned_fsdp_buffer_hook()
    hook_count = 0
    
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    for name, mod in fsdp_model.named_modules():
        if isinstance(mod, FSDP):
            mod.register_forward_pre_hook(pre_hook)
            mod.register_forward_hook(post_hook)
            hook_count += 1
    
    if trainer.device_mesh.get_rank() == 0:
        print(f"[Ascend] Applied 512-byte alignment via forward hooks on {hook_count} FSDP modules")


# =============================================================================
# Diagnostic: check alignment of all FSDP flat params
# =============================================================================

def check_fsdp_alignment(
    fsdp_model: nn.Module,
    alignment_bytes: int = ASCEND_ALIGNMENT_BYTES,
    verbose: bool = True,
) -> bool:
    """
    Check if all FSDP flat parameters are 512-byte aligned.
    
    Args:
        fsdp_model: FSDP-wrapped model.
        alignment_bytes: Required alignment.
        verbose: Print details for each misaligned parameter.
    
    Returns:
        True if all parameters are aligned, False otherwise.
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    
    all_aligned = True
    
    for name, mod in fsdp_model.named_modules():
        if isinstance(mod, FSDP):
            for handle in getattr(mod, '_handles', []):
                fp = handle.flat_param
                if fp is None:
                    continue
                
                ptr = fp.data_ptr()
                numel = fp.numel()
                byte_size = numel * fp.element_size()
                
                ptr_ok = (ptr % alignment_bytes) == 0
                size_ok = (byte_size % alignment_bytes) == 0
                
                if not (ptr_ok and size_ok):
                    all_aligned = False
                    if verbose:
                        align_elems = get_alignment_elements(fp.dtype, alignment_bytes)
                        print(
                            f"[Ascend WARN] {name}: "
                            f"ptr_aligned={ptr_ok}, size_aligned={size_ok}, "
                            f"numel={numel}, need_pad={align_numel(numel, fp.dtype) - numel}, "
                            f"dtype={fp.dtype}"
                        )
    
    if all_aligned and verbose:
        print("[Ascend] All FSDP flat parameters are 512-byte aligned.")
    
    return all_aligned


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
