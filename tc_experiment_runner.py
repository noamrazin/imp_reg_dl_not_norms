import argparse

import common.utils.logging as logging_utils
from trainers.tensor_completion_configurable_trainer import TensorCompletionConfigurableTrainer


def main():
    p = argparse.ArgumentParser()

    p.add_argument("-experiment_name", type=str, default="tensor_comp", help="Name of current experiment")
    p.add_argument("-dataset_path", type=str, default="data/tc/tensor_rank_1_fro_1_order_3_dim_8.pt",
                   help="Path to the the dataset file. If non given, will generate a new one.")
    p.add_argument("-num_samples", type=int, default=300, help="Number of samples to train on.")

    p.add_argument("-random_seed", type=int, default=-1, help="Initial random seed")
    p.add_argument("-output_dir", type=str, default="outputs/tc/r1", help="Directory to create the experiment output in.")
    p.add_argument("-file_log", action='store_true', help="Use file logging or console logging if false")
    p.add_argument("-store_checkpoints", action='store_true', help="Store checkpoints of the trainer during training")
    p.add_argument("-plot_metrics", action='store_true', help="Plot scalar metric values using matplotlib")
    p.add_argument("-disable_gpu", action='store_true', help="Disable gpu usage")
    p.add_argument("-gpu_id", type=int, default=0, help="Cuda gpu id to use")
    p.add_argument("-trainer_checkpoint", type=str, default="",
                   help="Path to trainer checkpoint to continue training with")
    p.add_argument("-epochs", type=int, default=1000000, help="Number of training epochs")
    p.add_argument("-validate_every", type=int, default=100, help="Run validation every this number of epochs")
    p.add_argument("-save_every_num_val", type=int, default=5, help="Saves checkpoints and plots every this number of validations.")
    p.add_argument("-lr", type=float, default=1e-2, help="Training learning rate")
    p.add_argument("-use_adaptive_lr", action="store_true", help="Uses an adaptive learning rate which scales the learning rate by dividing it by "
                                                                 "the sqrt of an exponentially moving average of square gradient norms.")
    p.add_argument("-init_std", type=float, default=1e-3, help="Init std for gaussian init")

    p.add_argument("-stop_on_zero_loss_tol", type=float, default=1e-6,
                   help="Stops when train loss reaches below this threshold.")

    p.add_argument("-tensor_rank_tol", type=float, default=1e-6, help="Tolerance threshold for computing tensor CP rank."
                                                                      "The rank will be the minimal rank that achieves an approximation lower than tol.")
    p.add_argument("-compute_final_tensor_ranks", action="store_true", help="Computes and stores the estimated cp tensor rank.")
    p.add_argument("-track_tensor_rank", action="store_true",
                   help="Track tensor cp rank throughout training - can computationaly expensive for larger input sizes.")

    args = p.parse_args()

    configurable_trainer = TensorCompletionConfigurableTrainer()
    configurable_trainer.fit(args.__dict__)


if __name__ == "__main__":
    logging_utils.init_console_logging()
    main()
