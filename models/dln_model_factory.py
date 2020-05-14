from models.deep_linear_net import DeepLinearNet


class DLNModelFactory:

    @staticmethod
    def create_same_dim_deep_linear_network(input_dim: int, output_dim: int, depth: int, weight_init_type: str = "normal",
                                            alpha: float = 1e-7, use_balanced_init: bool = False, enforce_pos_det: bool = False,
                                            enforce_neg_det: bool = False):
        hidden_dims = [min(input_dim, output_dim)] * (depth - 1)
        return DeepLinearNet(input_dim, output_dim, hidden_dims=hidden_dims, weight_init_type=weight_init_type,
                             alpha=alpha, use_balanced_init=use_balanced_init, enforce_pos_det=enforce_pos_det, enforce_neg_det=enforce_neg_det)
