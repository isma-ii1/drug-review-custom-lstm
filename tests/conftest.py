import torch

# Small matrix tests become much slower with many BLAS threads.
torch.set_num_threads(1)
