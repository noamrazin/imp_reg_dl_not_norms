import torch

import common.utils.tensor as tensor_utils
from common.evaluation.metrics.metric import AveragedMetric


class MatrixNormMetric(AveragedMetric):

    def __init__(self, norm: str = "fro"):
        super().__init__()
        self.norm = norm

    def _calc_metric(self, matrix: torch.Tensor):
        return torch.norm(matrix, p=self.norm).item(), 1


class MatrixRankMetric(AveragedMetric):

    def _calc_metric(self, matrix: torch.Tensor):
        return torch.matrix_rank(matrix).item(), 1


class MatrixEffectiveRankMetric(AveragedMetric):

    def _calc_metric(self, matrix: torch.Tensor):
        effective_rank = tensor_utils.matrix_effective_rank(matrix)
        return effective_rank, 1


class ReconstructionErrorMetric(AveragedMetric):

    def __init__(self, normalized=False):
        super().__init__()
        self.normalized = normalized

    def _calc_metric(self, tensor: torch.Tensor, target_tensor: torch.Tensor):
        reconstruction_error = compute_reconstruction_error(tensor, target_tensor, self.normalized)
        return reconstruction_error, 1


def compute_reconstruction_error(tensor, target_tensor, normalize=False):
    reconstruction_error = torch.norm(tensor - target_tensor, p="fro").item()
    if normalize:
        target_norm = torch.norm(target_tensor, p="fro").item()
        if target_norm != 0:
            reconstruction_error /= target_norm

    return reconstruction_error


class MatrixEntryMetric(AveragedMetric):

    def __init__(self, i, j):
        super().__init__()
        self.i = i
        self.j = j

    def _calc_metric(self, matrix: torch.Tensor):
        return matrix[self.i, self.j].item(), 1
