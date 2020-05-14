import itertools

import torch
import torch.utils.data


class TensorCompletionDataset(torch.utils.data.Dataset):

    def __init__(self, target_tensor, target_cp_rank, train_indices_order):
        self.target_tensor = target_tensor
        self.target_cp_rank = target_cp_rank
        self.train_indices_order = train_indices_order
        self.all_indices_tensor = self.__create_all_indices_tensor(list((target_tensor.size())))

    def __create_all_indices_tensor(self, mode_dims):
        indices = []
        per_mode_options = [range(dim) for dim in mode_dims]
        for tensor_index in itertools.product(*per_mode_options):
            indices.append(torch.tensor(tensor_index, dtype=torch.long))

        return torch.stack(indices)

    def __getitem__(self, index: int):
        tensor_indices = self.all_indices_tensor[index]
        return tensor_indices, self.target_tensor[tuple(tensor_indices.tolist())]

    def __len__(self) -> int:
        return self.target_tensor.numel()

    def save(self, path: str):
        state_dict = {
            "target_tensor": self.target_tensor,
            "target_cp_rank": self.target_cp_rank,
            "train_indices_order": self.train_indices_order,
        }
        torch.save(state_dict, path)

    @staticmethod
    def load(path: str):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        return TensorCompletionDataset(state_dict["target_tensor"], state_dict["target_cp_rank"], state_dict["train_indices_order"])
