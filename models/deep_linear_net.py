from typing import Sequence

import torch
import torch.nn as nn


class DeepLinearNet(nn.Module):
    """
    Deep Linear Network. This is simply a fully connected network with identity activations.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Sequence[int] = None, weight_init_type: str = "normal",
                 alpha: float = 1e-7, use_balanced_init: bool = False, enforce_pos_det: bool = False, enforce_neg_det: bool = False):
        """
        :param input_dim: Number of input dimensions.
        :param output_dim: Number of output dimensions.
        :param hidden_dims: Sequence of hidden dimensions.
        :param weight_init_type: Str code for type of initialization. Supports: 'normal', 'identity'.
        :param alpha: Standard deviation/multiplicative factor of the initialization. The layers will be initialized such that this will be the
        standard deviation of the product matrix (or multiplicative scale for identity init).
        :param use_balanced_init: Balanced initialization - product matrix initialized with identity and normal and then factor matrices are initialized
        such they are balanced (and multiplying them results in the product matrix).
        :param enforce_pos_det: Initialize such that the product matrix has a positive determinant. In random initialization, it will init multiple
        times until a matrix with positive determinant is obtained.
        :param enforce_neg_det: Initialize such that the product matrix has a negative determinant. In random initialization, it will init multiple
        times until a matrix with negative determinant is obtained.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims if hidden_dims is not None else []
        self.depth = len(hidden_dims) + 1
        self.weight_init_type = weight_init_type
        self.alpha = alpha
        self.use_balanced_init = use_balanced_init
        self.enforce_pos_det = enforce_pos_det
        self.enforce_neg_det = enforce_neg_det
        self.__verify_params()

        self.layers = nn.ParameterList(self.__create_layers())

    def __verify_params(self):
        if self.enforce_pos_det and self.enforce_neg_det:
            raise ValueError("Parameters 'enforce_pos_det' and 'enforce_neg_det' are mutually exclusive.")

    def __initialize_new_layer(self, input_dim: int, output_dim: int):
        if self.weight_init_type == "normal":
            weights = torch.randn(input_dim, output_dim) * self.alpha
            return nn.Parameter(weights, requires_grad=True)

        if self.weight_init_type == "identity":
            weights = self.alpha * torch.eye(input_dim, output_dim, dtype=torch.float)
            return nn.Parameter(weights, requires_grad=True)

        raise ValueError(f"Unsupported weight initialization type: {self.weight_init_type}")

    def __create_layers(self):
        if self.use_balanced_init:
            return self.__create_balanced_init_layers()

        layers = []
        curr_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(self.__initialize_new_layer(curr_dim, hidden_dim))
            curr_dim = hidden_dim

        layers.append(self.__initialize_new_layer(curr_dim, self.output_dim))

        if self.enforce_pos_det and not self.__prod_mat_has_pos_det(layers):
            return self.__create_layers()
        elif self.enforce_neg_det and self.__prod_mat_has_pos_det(layers):
            return self.__create_layers()

        return layers

    def __prod_mat_has_pos_det(self, layers):
        with torch.no_grad():
            prod_mat = self.__compute_prod_matrix_from_layers(layers)
            return torch.det(prod_mat).item() >= 0

    def __create_balanced_init_layers(self):
        prod_mat = self.__initialize_new_layer(self.input_dim, self.output_dim)
        if self.depth == 1:
            return [prod_mat]

        U, S, V = torch.svd(prod_mat.data)
        S_sqrt_N = S.pow(1 / self.depth)

        layers = [nn.Parameter(torch.matmul(U, self.__init_diag_matrix(self.input_dim, self.hidden_dims[0], S_sqrt_N)), requires_grad=True)]
        curr_dim = self.hidden_dims[0]
        for hidden_dim in self.hidden_dims[1:]:
            layers.append(nn.Parameter(self.__init_diag_matrix(curr_dim, hidden_dim, S_sqrt_N)))
            curr_dim = hidden_dim

        layers.append(nn.Parameter(torch.matmul(self.__init_diag_matrix(curr_dim, V.size(1), S_sqrt_N), V.t()), requires_grad=True))
        return layers

    def __init_diag_matrix(self, rows, cols, diag_elements):
        mat = torch.zeros(rows, cols)
        mat[range(len(diag_elements)), range(len(diag_elements))] = diag_elements
        return mat

    def compute_prod_matrix(self):
        return self.__compute_prod_matrix_from_layers(self.layers)

    def __compute_prod_matrix_from_layers(self, layers):
        curr = layers[0]
        for i in range(1, len(layers)):
            curr = torch.matmul(curr, layers[i])

        return curr

    def forward(self, x):
        prod_mat = self.compute_prod_matrix()
        return torch.matmul(x, prod_mat)
