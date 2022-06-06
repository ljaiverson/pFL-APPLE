import torch
import random

import numpy as np

import os
import argparse

########################################### parser and creating paths ###########################################
def create_parser():
    """Creates a parser for command-line arguments.
    """
    parser = argparse.ArgumentParser(add_help=True, description='Adaptive Personalized Cross-Silo Federated Learning (APPLE)')

    # Random seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Data sources
    parser.add_argument("--data", type=str, default="mnist",  choices=["mnist", "cifar10", "pathmnist", "organmnist_axial"], help="Dataset name")
    parser.add_argument("--distribution", type=str, default="non-iid-pathological", choices=["non-iid-practical", "non-iid-pathological"], help="The type of non-IID")

    # General training hyper-parameters
    parser.add_argument("--image_size", type=int, default=28, help="The image size")
    parser.add_argument("--batch_size", type=int, default=256, help="The number of images in a batch")
    parser.add_argument("--num_workers", type=int, default=0, help="The number of threads to use for the DataLoader")
    parser.add_argument("--num_rounds", type=int, default=160, help="The number of training rounds")
    parser.add_argument("--num_local_epochs", type=int, default=5, help="The number of local training epochs")
    parser.add_argument("--decay", type=float, default=1.0, help="The learning rate decay per round")
    
    # APPLE optimization
    parser.add_argument("--lr_net", type=float, default=0.01, help="The learning rate for the core models")
    parser.add_argument("--lr_coef", type=float, default=0.001, help="The learning rate for the Directed Relationships (ps)")
    
    # APPLE regularizer
    parser.add_argument("--mu", type=float, default=0.1, help="The coeficient for the regularizer")
    parser.add_argument("--reg_last_round_over_total_round", type=float, default=0.1, help="reg_last_round / total_rounds.")
    parser.add_argument("--scheduler", type=str, default="exponential", choices=["cosine", "exponential"], help="The shape of the loss scheduler")

    # APPLE limited bandwidth
    parser.add_argument("--limit_downloaded_models", type=int, default=10000, help="The maximum number of core models each client can download in a round")
    parser.add_argument("--lamb", type=float, default=1.0, help="The coeficient for the limited bandwidth")
    parser.add_argument("--thresh", type=float, default=1.5, help="The threshold of base for the limited bandwidth")

    # Results
    parser.add_argument("--log_step", type=int , default=10, help="Output every X batches")
    parser.add_argument("--data_dist_dir", type=str, default="./data_dist", help="Directory of the data distribution figure")
    parser.add_argument("--hist_dir", type=str, default="./results", help="Directory of the results")

    args = parser.parse_args()
    return args

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_dir(args):
    if not os.path.exists(args.data_dist_dir):
        os.makedirs(args.data_dist_dir)
    if not os.path.exists(args.hist_dir):
        os.makedirs(args.hist_dir)
