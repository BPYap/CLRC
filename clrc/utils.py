import torch


def calculate_train_steps(pl_module):
    trainer = pl_module.trainer
    num_devices = max(1, trainer.num_gpus, trainer.num_processes)
    if trainer.tpu_cores:
        num_devices = max(num_devices, trainer.tpu_cores)

    steps_per_epoch = (len(pl_module.train_dataloader()) // num_devices) // trainer.accumulate_grad_batches
    total_steps = steps_per_epoch * trainer.max_epochs
    if trainer.max_steps is not None:
        total_steps = min(total_steps, trainer.max_steps)

    return steps_per_epoch, total_steps


def is_distributed():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class SyncFunction(torch.autograd.Function):
    """Adapted from https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/
    simclr/simclr_module.py#L20"""

    @staticmethod
    def forward(ctx, tensor):
        ctx.batch_size = tensor.shape[0]

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor)
        gathered_tensor = torch.cat(gathered_tensor, 0)

        return gathered_tensor

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        torch.distributed.all_reduce(grad_input, op=torch.distributed.ReduceOp.SUM, async_op=False)

        idx_from = torch.distributed.get_rank() * ctx.batch_size
        idx_to = (torch.distributed.get_rank() + 1) * ctx.batch_size

        return grad_input[idx_from:idx_to]


def gather_tensors(tensors):
    if is_distributed():
        tensors = SyncFunction.apply(tensors)

    return tensors
