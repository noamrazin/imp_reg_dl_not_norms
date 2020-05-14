from typing import List, Union

import numpy as np
import torch
import torch.nn as nn

import common.utils.tensor as tensor_utils


class TensorCPFactorization(nn.Module):
    """
    Tensor CP factorization model.
    """

    def __init__(self, num_dim_per_mode: Union[int, List[int]], order: int, rank: int = -1, init_std=0.01):
        """
        :param num_dim_per_mode: Number of dimensions per tensor mode. If int then will use same dim for all orders, otherwise must be a list of
        length of the order with the dimension for each mode.
        :param order: Order of the tensor.
        :param rank: Number of tensor products sums of the factorization. If -1 the default is the max possible CP rank which is the product
        of all dimensions except the max.
        :param init_std: std of vectors gaussian init.
        """
        super().__init__()
        self.num_dim_per_mode = [num_dim_per_mode] * order if isinstance(num_dim_per_mode, int) else num_dim_per_mode
        self.order = order
        self.rank = rank if rank != -1 else self.__compute_max_possible_tensor_rank()
        self.init_std = init_std

        self.factors = nn.ParameterList(self.__create_factors())

    def __compute_max_possible_tensor_rank(self):
        tensor_dims = list(self.num_dim_per_mode)
        max_index = tensor_dims.index(max(tensor_dims))
        tensor_dims.pop(max_index)

        return np.prod(tensor_dims)

    def __create_factors(self):
        factors = []
        for dim in self.num_dim_per_mode:
            factor = torch.randn(dim, self.rank) * self.init_std
            factors.append(nn.Parameter(factor, requires_grad=True))

        return factors

    def compute_tensor(self):
        return tensor_utils.reconstruct_parafac(self.factors)

    def forward(self, tensor_of_indices):
        tensor = self.compute_tensor()
        split_indices = [tensor_of_indices[:, i] for i in range(tensor_of_indices.size(1))]
        tensor_values = tensor[split_indices]
        return tensor_values
