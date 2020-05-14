import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

import common.utils.logging as logging_utils
import common.utils.tensor as tensor_utils
from datasets.tensor_completion_dataset import TensorCompletionDataset


def __set_initial_random_seed(random_seed: int):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def create_tensor_completion_dataset(target_tensor_cp_rank, target_fro_norm, mode_dim_size, order):
    target_tensor = tensor_utils.create_tensor_with_cp_rank([mode_dim_size] * order, target_tensor_cp_rank, fro_norm=target_fro_norm)
    train_indices_order = torch.randperm(target_tensor.numel()).tolist()
    dataset = TensorCompletionDataset(target_tensor, target_tensor_cp_rank, train_indices_order)
    return dataset


def create_and_save_dataset(args):
    dataset = create_tensor_completion_dataset(args.target_tensor_cp_rank, args.target_fro_norm, args.mode_dim_size, args.order)

    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
    if args.custom_file_name:
        file_name = f"{args.custom_file_name}_{now_utc_str}.pt"
    else:
        file_name = f"tensor_rank_{args.target_tensor_cp_rank}_fro_{int(args.target_fro_norm)}" \
            f"_order_{args.order}_dim_{args.mode_dim_size}_{now_utc_str}.pt"

    dataset.save(os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-random_seed", type=int, default=-1, help="Initial random seed")
    p.add_argument("-output_dir", type=str, default="data/tc", help="Path to the directory to save the target matrix and dataset at.")
    p.add_argument("-custom_file_name", type=str, default="", help="Custom file name prefix for the dataset.")

    p.add_argument("-target_tensor_cp_rank", type=int, default=1,
                   help="CP rank of the target tensor. Use -1 for no rank constraint (tensor will be generated randomly)")
    p.add_argument("-target_fro_norm", type=float, default=1.0, help="Fro norm of the target tensor.")

    p.add_argument("-mode_dim_size", type=int, default=10, help="Number of dimensions per each mode.")
    p.add_argument("-order", type=int, default=3, help="Order of the tensor (number of modes).")

    args = p.parse_args()

    logging_utils.init_console_logging()
    __set_initial_random_seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    create_and_save_dataset(args)
