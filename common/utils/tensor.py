import string

import torch


def to_numpy(tensor):
    """
    Converts tensor to numpy ndarray. Will move tensor to cpu and detach it before converison. Numpy ndarray will share memory
    of the tensor.
    :param tensor: input pytorch tensor.
    :return: numpy ndarray with shared memory of the given tensor.
    """
    return tensor.cpu().detach().numpy()


def matrix_effective_rank(matrix):
    """
    Calculates the effective rank of the matrix.
    :param matrix: torch matrix of size (N, M)
    :return: Effective rank of the matrix.
    """
    svd_result = torch.svd(matrix, compute_uv=False)
    singular_values = svd_result.S
    non_zero_singular_values = singular_values[singular_values != 0]
    normalized_non_zero_singular_values = non_zero_singular_values / non_zero_singular_values.sum()

    singular_values_entropy = -(normalized_non_zero_singular_values * torch.log(normalized_non_zero_singular_values)).sum()
    return torch.exp(singular_values_entropy).item()


def create_tensor_with_cp_rank(num_dim_per_mode, cp_rank, fro_norm=1):
    """
    Creates a tensor with the given CP rank.
    :param num_dim_per_mode: List of dimensions per mode.
    :param cp_rank: Desired CP rank (will determine number size of factors per mode).
    :param fro_norm: Frobenius norm of tensor (the tensor is normalized at the end to this nor m).
    :return: Tensor of the given CP rank and Frobenius norm.
    """
    if cp_rank == -1:
        tensor = torch.randn(*num_dim_per_mode)
    else:
        factors = []
        for dim in num_dim_per_mode:
            factor = torch.randn(dim, cp_rank)
            factors.append(factor)

        tensor = reconstruct_parafac(factors)

    tensor = (tensor / torch.norm(tensor, p="fro")) * fro_norm
    return tensor


# Adaptation of https://stackoverflow.com/a/13772838/160466
def reconstruct_parafac(factors):
    """
    Reconstructs a tensor from its parafac decomposition. Each factor i is of size (d_i, r), and the tensor is created by computing the tensor
    product for the corresponding vectors in each factor and summing all r r tensor products.
    :param factors: List of tensors of size (d_i, r).
    :return: Tensor of size (d_1,...,d_n)
    """
    ndims = len(factors)
    request = ''
    for temp_dim in range(ndims):
        request += string.ascii_lowercase[temp_dim] + 'z,'

    request = request[:-1] + '->' + string.ascii_lowercase[:ndims]
    return torch.einsum(request, *factors)
