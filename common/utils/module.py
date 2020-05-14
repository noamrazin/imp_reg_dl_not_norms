import torch
import torch.nn as nn
import torch.utils.data


def get_number_of_parameters(module: nn.Module) -> int:
    """
    Returns the number of parameters in the module.
    :param module: PyTorch Module.
    :return: Number of parameters in the module.
    """
    return sum(p.numel() for p in module.parameters())


def get_use_gpu(disable_gpu: bool = False) -> bool:
    """
    Returns true if cuda is available and no explicit disable cuda flag given.
    """
    return torch.cuda.is_available() and not disable_gpu


def get_device(disable_gpu: bool = False, cuda_id: int = 0):
    """
    Returns a gpu cuda device if available and cpu device otherwise.
    """
    if get_use_gpu(disable_gpu):
        return torch.device(f"cuda:{cuda_id}")
    return torch.device("cpu")


def set_requires_grad(module: nn.Module, requires_grad: bool):
    """
    Sets the requires grad flag for all of the modules parameters.
    :param module: pytorch module.
    :param requires_grad: requires grad flag value.
    """
    for param in module.parameters():
        param.requires_grad = requires_grad
