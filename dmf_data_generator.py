import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch

import common.utils.logging as logging_utils
from datasets.matrix_sensing_dataset import MatrixSensingDataset


def __create_custom_target_matrix(num_rows, num_cols):
    mat = torch.zeros(num_rows, num_cols, dtype=torch.float)
    mat[0, 1] = 1
    mat[1, 0] = 1

    # In case size is d x d for d > 2
    for i in range(2, min(num_rows, num_cols)):
        mat[i, i] = 1

    return mat


def __create_custom_sensing_matrices(num_rows, num_cols):
    sensing_matrices = []
    for i in range(num_rows):
        for j in range(num_cols):
            if i == 0 and j == 0:  # (0,0)
                continue
            sensing_mat = torch.zeros(num_rows, num_cols, dtype=torch.float)
            sensing_mat[i, j] = 1
            sensing_matrices.append(sensing_mat)

    return torch.stack(sensing_matrices)


def __set_initial_random_seed(random_seed: int):
    if random_seed != -1:
        np.random.seed(random_seed)
        torch.random.manual_seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)


def create_target_matrix(num_rows, num_cols, rank, use_symmetric=False, use_custom_target=False):
    if use_custom_target:
        return __create_custom_target_matrix(num_rows, num_cols)

    U = np.random.randn(num_rows, rank).astype(np.float32)
    V = U if use_symmetric else np.random.randn(num_cols, rank).astype(np.float32)

    W_target = U.dot(V.T)
    W_target = (W_target / np.linalg.norm(W_target, 'fro')) * num_rows

    target_sv = np.linalg.svd(W_target, compute_uv=False)
    logging_utils.info("Created target matrix. Singular values = %s, Fro(w) = %.3f", target_sv[:rank], np.linalg.norm(W_target, ord='fro'))
    return torch.from_numpy(W_target)


def create_matrix_sensing_dataset(args, target_matrix):
    if args.use_custom_sensing_matrices:
        A = __create_custom_sensing_matrices(args.num_rows, args.num_cols)
    elif args.task_type == "sensing":
        A = torch.randn(args.num_samples, args.num_rows, args.num_cols) * args.sensing_std
    elif args.task_type == "completion":
        indices = torch.multinomial(torch.ones(args.num_rows * args.num_cols), args.num_samples, replacement=False)
        us, vs = indices // args.num_rows, indices % args.num_cols

        A = torch.zeros(args.num_samples, args.num_rows, args.num_cols)
        A[range(args.num_samples), us, vs] = 1

    else:
        raise ValueError(f"Unsupported task type: {args.task_type}.")

    y = (A * target_matrix.unsqueeze(0)).sum(dim=(1, 2))
    return MatrixSensingDataset(A, y, target_matrix)


def create_and_save_dataset(args):
    target_matrix = create_target_matrix(args.num_rows, args.num_cols, args.rank, args.use_symmetric_target, args.use_custom_target)
    dataset = create_matrix_sensing_dataset(args, target_matrix)

    now_utc_str = datetime.utcnow().strftime("%Y_%m_%d-%H_%M_%S")
    if args.custom_file_name:
        file_name = f"{args.custom_file_name}_{now_utc_str}.pt"
    else:
        symm_suffix = "_sym_" if args.use_symmetric_target else "_"
        file_name = f"{args.task_type}_rank_{args.rank}_{args.num_rows}x{args.num_cols}{symm_suffix}{now_utc_str}.pt"

    dataset.save(os.path.join(args.output_dir, file_name))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-custom_file_name", type=str, default="", help="Custom file name prefix for the dataset.")
    p.add_argument("-random_seed", type=int, default=-1, help="Initial random seed")
    p.add_argument("-output_dir", type=str, default="data/dmf", help="Path to the directory to save the target matrix and dataset at.")
    p.add_argument("-num_samples", type=int, default=2500, help="Number of sensing samples to create.")
    p.add_argument("-num_rows", type=int, default=50, help="Number of rows of the matrix to use for sensing.")
    p.add_argument("-num_cols", type=int, default=50, help="Number of rows of the matrix to use for sensing.")
    p.add_argument("-rank", type=int, default=5, help="Rank of the random matrix to use for sensing.")
    p.add_argument("-use_symmetric_target", action="store_true", help="Use symmetric target matrix.")
    p.add_argument("-use_custom_target", action="store_true", help="Use custom target matrix.")
    p.add_argument("-use_custom_sensing_matrices", action="store_true", help="Use custom sensing matrices.")
    p.add_argument("-task_type", type=str, default="sensing", help="Task type. Can be either 'sensing' for random matrix sensing or 'completion' "
                                                                   "for matrix completion")
    p.add_argument("-sensing_std", type=float, default=1, help="Standard deviation of gaussian sensing matrices when task type is 'sensing'.")

    args = p.parse_args()

    logging_utils.init_console_logging()
    __set_initial_random_seed(args.random_seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    create_and_save_dataset(args)
