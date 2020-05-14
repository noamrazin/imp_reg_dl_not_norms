import torch
import torch.utils.data


class MatrixSensingDataset(torch.utils.data.Dataset):

    def __init__(self, A, y, target_matrix):
        self.A = A
        self.y = y
        self.target_matrix = target_matrix

    def __getitem__(self, index: int):
        return self.A[index], self.y[index]

    def __len__(self) -> int:
        return self.A.size(0)

    def save(self, path: str):
        state_dict = {
            "A": self.A,
            "y": self.y,
            "target_matrix": self.target_matrix
        }
        torch.save(state_dict, path)

    @staticmethod
    def load(path: str):
        state_dict = torch.load(path, map_location=torch.device("cpu"))
        return MatrixSensingDataset(state_dict["A"], state_dict["y"], state_dict["target_matrix"])
