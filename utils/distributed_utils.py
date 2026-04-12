import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def _get_attr(args, name, default=None):
    return getattr(args, name, default)


def is_distributed_training(args):
    return bool(_get_attr(args, 'distributed_training', False))


def get_rank(args=None):
    if args is not None and hasattr(args, 'rank'):
        return int(args.rank)
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size(args=None):
    if args is not None and hasattr(args, 'world_size'):
        return int(args.world_size)
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process(args=None):
    return get_rank(args) == 0


def setup_distributed(args, rank, world_size):
    if not torch.cuda.is_available():
        raise RuntimeError('DDP distributed training requires CUDA GPUs, but CUDA is unavailable.')

    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    os.environ.setdefault('MASTER_PORT', '29500')

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

    args.rank = rank
    args.local_rank = rank
    args.world_size = world_size
    args.device = f'cuda:{rank}'
    return args


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def maybe_wrap_ddp(model, args):
    if not is_distributed_training(args):
        return model

    device = torch.device(args.device)
    device_index = 0 if device.index is None else device.index
    return DDP(
        model,
        device_ids=[device_index],
        output_device=device_index,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )


def unwrap_distributed_model(model):
    if isinstance(model, DDP):
        return model.module
    return model


def reduce_tensor_sum(value, device):
    tensor = value if torch.is_tensor(value) else torch.tensor(value, dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def reduce_metrics(loss_sum, correct, total, num_batches, device):
    packed = torch.tensor(
        [float(loss_sum), float(correct), float(total), float(num_batches)],
        dtype=torch.float64,
        device=device,
    )
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    return packed.tolist()


def get_model_state_dict(model, args):
    return unwrap_distributed_model(model).state_dict()
