###############################################################################
## Description: Define the function for generating the dataset for the k-bit ##
##              parity problem for a given k                                 ##
###############################################################################

import torch
from torch import Tensor
from typing import Tuple

def generate_k_bit_parity_dataset(k: int) -> Tuple[Tensor, Tensor]:
    """
        Description: Generates complete dataset of input vector,
                     label pairs (X, y) for the k-bit parity problem
        Args:
            k (int): Length of bit string (e.g, k=3 for 3-bit parity problem)
        Returns:
            parity_dataset (X, y): dataset containing the
                           complete set of (input vector, label) pairs defining the
                           k-bit parity problem
    """
    # Calculate all k-bit strings and store in tensor X
    decimal_values = torch.arange(2**k).unsqueeze(1)
    bit_positions = torch.arange(k - 1, -1, -1)
    binary_strings = ((decimal_values >> bit_positions) & 1)
    X = binary_strings.to(torch.float32)

    # Create labels storing the parity (even/odd) of each bit string
    parity_tensor = (X.sum(dim=1) % 2).unsqueeze(1)
    y = parity_tensor.to(torch.float32)

    parity_dataset = (X, y)
    return parity_dataset