import torch
import numpy as np

def reg_loss(p, p0, coef):
    l2_norm = lambda a: torch.sum(a * a)
    return l2_norm(p - p0) * coef

def get_reg_coef(args, r):
    if args.mu == 0:
        return 0
    num_dynamic_rounds = int(args.num_rounds * args.reg_last_round_over_total_round)
    if r >= num_dynamic_rounds:
        return 0
    if args.scheduler == "cosine":
        # cosine
        phase = np.pi / num_dynamic_rounds * r
        return (np.cos(phase) + 1) / 2 * args.mu
    if args.scheduler == "exponential":
        # exponential
        epsilon = 0.001
        return ((epsilon / 1) ** (r / num_dynamic_rounds)) * args.mu