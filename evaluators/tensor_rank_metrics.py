import string
from functools import reduce

import numpy as np
import torch

import common.utils.tensor as tensor_utils
from common.evaluation.metrics import ScalarMetric


class ModelTensorCPRank(ScalarMetric):

    def __init__(self, tol=1e-6):
        """
        :param tol: Tolerance threshold used to find the rank of the tensor. The rank will be the minimal Parafac decomposition that has a
        reconstruction mse of less than tol.
        """
        self.tol = tol
        self.__current_value = None

    def __call__(self, tensor):
        cp_rank = find_cp_rank(tensor, tol=self.tol)
        self.__current_value = cp_rank
        return cp_rank

    def current_value(self):
        return self.__current_value

    def has_epoch_metric_to_update(self) -> bool:
        return self.__current_value is not None

    def reset_current_epoch_values(self):
        self.__current_value = None


def find_cp_rank(tensor, max_rank=-1, tol=1e-6):
    """
    Finds the CP rank of the tensor by searching for the minimal r for which the reconstruction of the parafac decomposition is less than tol.
    :param tensor: PyTorch Tensor to compute CP rank for.
    :param max_rank: max rank to check. By default, will check for all ranks up to the product of dimensions except the maximal.
    :param tol: tolerance threshold used to determine the cp rank. The rank is determined as the minimal rank of CP decomposition for which
    the mse of the reconstruction is below tol.
    :return: estimated cp_rank of tensor. Returns -1 on failure.
    """
    if torch.allclose(tensor, torch.zeros_like(tensor)):
        return 0

    max_rank = max_rank if max_rank != -1 else __compute_max_possible_tensor_rank(tensor)

    first = 0
    last = max_rank - 1
    curr_min = max_rank
    try:
        while first <= last:
            mid = (first + last) // 2
            mse = compute_reconstruction_mse(tensor, r=mid + 1, tol=tol)
            if mse < tol:
                curr_min = mid + 1
                last = mid - 1
            else:
                first = mid + 1
    except np.linalg.LinAlgError:
        return -1

    return curr_min.item()


def compute_reconstruction_mse(tensor, r, tol=1e-6):
    coeff, factors = cp_als(tensor, r, tol=tol)
    factors[0] = factors[0] * coeff
    reconstructed_tensor = tensor_utils.reconstruct_parafac(factors)

    mse = ((tensor - reconstructed_tensor) ** 2).sum() / tensor.numel()
    return mse.item()


##########################################################################################
# Adapted from https://www.kaggle.com/nicw102168/rank-of-random-2x2x2-tensors ############
##########################################################################################
def cp_als(tensor, r, max_iter=1000, validate_convergence_every=1, tol=1e-6):
    np_tensor = tensor.detach().numpy()
    factors = [np.random.randn(dim, r) for dim in np_tensor.shape]
    num_modes = len(np_tensor.shape)

    for i in range(max_iter):
        for n in range(num_modes):
            other_modes = [m for m in range(num_modes) if m != n]
            other_factors = [factors[m] for m in other_modes]

            V_elements = [np.matmul(factor.T, factor) for factor in other_factors]
            V = reduce(lambda a, b: a * b, V_elements)
            W = reduce(__khatri_rao_product, other_factors)
            Xn = np.rollaxis(np_tensor, n).reshape(np_tensor.shape[n], -1)

            update = np.matmul(Xn, np.matmul(W, np.linalg.pinv(V)))

            coeff = np.linalg.norm(factors[n], axis=0, keepdims=True)
            factors[n] = update / coeff

        if i % validate_convergence_every == 0:
            curr_factors = [factors[0] * coeff] + factors[1:]
            reconstructed_tensor = np_reconstruct_parafac(curr_factors)
            mse = ((np_tensor - reconstructed_tensor) ** 2).sum() / np_tensor.size
            if mse < tol:
                break

    factors = [torch.from_numpy(factor) for factor in factors]
    coeff = torch.from_numpy(coeff)
    return coeff, factors


def __khatri_rao_product(A, B):
    return np.einsum("ij, kj -> ikj", A, B).reshape(-1, A.shape[1])


# Adaptation of https://stackoverflow.com/a/13772838/160466
def np_reconstruct_parafac(factors):
    ndims = len(factors)
    request = ''
    for temp_dim in range(ndims):
        request += string.ascii_lowercase[temp_dim] + 'z,'
    request = request[:-1] + '->' + string.ascii_lowercase[:ndims]
    return np.einsum(request, *factors)


def __compute_max_possible_tensor_rank(tensor):
    tensor_dims = list(tensor.size())
    max_index = tensor_dims.index(max(tensor_dims))
    tensor_dims.pop(max_index)

    return np.prod(tensor_dims)
