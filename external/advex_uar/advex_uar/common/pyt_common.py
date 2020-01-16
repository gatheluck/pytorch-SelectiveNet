import math

def get_step_size(epsilon, n_iters, use_max=False):
    if use_max:
        return epsilon
    else:
        return epsilon / math.sqrt(n_iters)