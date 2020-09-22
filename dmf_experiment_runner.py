import argparse

import common.utils.logging as logging_utils
from trainers.dln_matrix_sensing_configurable_trainer import DLNMatrixSensingConfigurableTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-experiment_name", type=str, default="matrix_comp", help="Name of current experiment.")
    p.add_argument("-dataset_path", type=str, default="data/dmf/completion_2x2_w_1_1_0.pt", help="Path to the matrix sensing dataset.")
    p.add_argument("-num_samples", type=int, default=3, help="Number of sensing samples to use.")

    p.add_argument("-random_seed", type=int, default=-1, help="Initial random seed.")
    p.add_argument("-output_dir", type=str, default="outputs/dmf", help="Directory to create the experiment output in.")
    p.add_argument("-file_log", action='store_true', help="Use file logging or console logging if false")
    p.add_argument("-store_checkpoints", action='store_true', help="Store checkpoints of the trainer during training.")
    p.add_argument("-plot_metrics", action='store_true', help="Plot scalar metric values using matplotlib.")
    p.add_argument("-save_results", action='store_true', help="Save final e2e matrix and metric values.")
    p.add_argument("-disable_gpu", action='store_true', help="Disable gpu usage")
    p.add_argument("-gpu_id", type=int, default=0, help="Cuda gpu id to use")

    p.add_argument("-depth", type=int, default=3, help="Depth of the factorization used to define the weight matrix.")
    p.add_argument("-weight_init_type", type=str, default="normal", help="Type of initialization for weights. "
                                                                         "Currently supports: 'identity', 'normal'")
    p.add_argument("-init_alpha", type=float, default=8e-3, help="Weight initialization std or identity multiplicative scalar (depending on init "
                                                                 "type) for each layer. If 'use_balanced_init' is used, the product matrix will be "
                                                                 "initialized using this value.")
    p.add_argument("-use_balanced_init", action='store_true', help="Initializes the factors such that they are balanced.")
    p.add_argument("-enforce_pos_det", action='store_true', help="Resamples initialization until product matrix has positive determinant.")
    p.add_argument("-enforce_neg_det", action='store_true', help="Resamples initialization until product matrix has negative determinant."
                                                                 " Mutually exclusive with 'enforce_pos_det'.")
    p.add_argument("-epochs", type=int, default=5000000, help="Number of training epochs")
    p.add_argument("-stop_on_zero_loss_tol", type=float, default=1e-4, help="Stops when train loss reaches below this threshold.")
    p.add_argument("-validate_every", type=int, default=50, help="Run validation every this number of epochs")
    p.add_argument("-save_every_num_val", type=int, default=5, help="Saves checkpoints and plots every this number of validations.")
    p.add_argument("-lr", type=float, default=3e-3, help="Training learning rate")
    p.add_argument("-tracked_e2e_value_of", type=int, nargs="+", action="append", default=[],
                   help="Indices of entries to track in the product matrix.")
    p.add_argument("-track_singular_values", action="store_true", help="Tracks the singular values of the product matrix.")

    args = p.parse_args()

    configurable_trainer = DLNMatrixSensingConfigurableTrainer()
    configurable_trainer.fit(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
