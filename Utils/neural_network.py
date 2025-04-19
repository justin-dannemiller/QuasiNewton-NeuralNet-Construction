###############################################################################
## Description: Defines the utility functions for constructing the neural    ##
##              network, forward propagation, backpropagation, and           ##
##              error/metric functions.                                      ##
###############################################################################

import numpy as np
import torch


###################### Weight packing/unpacking utilities #####################
def unpack_weights(W, n_features, h):
    """Return (Y, z) where
       Y shape = (n_features, h),  z shape = (h,)"""
    Y = W[:n_features * h].reshape(n_features, h)
    z = W[n_features * h : n_features * h + h]
    return Y, z

def pack_weights(Y, z):
    """Flatten (Y, z) back to a 1‑D vector."""
    return np.concatenate([Y.ravel(), z.ravel()])
###############################################################################


################ Network Architecture Construction & Updates ##################
def initialize_FC_neural_net(n_feats: int, n_hidden: int) -> np.ndarray:
    """
        Description: Initializes fully connected neural network for binary 
                     classification with n_feats input neurons and n_hidden
                     hidden neurons. Returns the initialized weights of both
                     hidden and output layer stacked in a single np array.
        Args:
            n_feats (int): Number of featurse in input vectors
            n_hidden (int): Number of units in hidden layer
        Returns
            W_init (np.ndarray): Array containing all initialized weights
    """
    y_init = torch.empty(n_feats, n_hidden).uniform_(-1, 1) # input-hidden weights
    z_init = torch.empty(n_hidden, 1).uniform_(-1, 1) # hidden-output weights
    W_init = torch.cat([y_init.flatten(), z_init.flatten()])
    return W_init

def add_neuron_to_network(W, n_feats, n_output_neurons, n_hidden):
    """
        Description: Adds additional neuron to hidden layer
    """
    random_init = (-1,1)
    y,z = unpack_weights(W, n_feats, n_hidden)
    y_add = np.random.uniform(*random_init, size=(n_feats, 1)) # n is n-dim vector sample
    z_add = np.random.uniform(*random_init, size=(1, n_output_neurons))

    y_new = np.concatenate([y, y_add], axis=1)
    z_new = np.concatenate([z, z_add], axis=0)

    W_new = pack_weights(y_new, z_new)
    n_hidden += 1

    return W_new, n_hidden

###############################################################################


######################## Forward & backward prop. #############################
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
  sigmoid_z = sigmoid(z)
  return sigmoid_z * (1-sigmoid_z)

def forward_pass(X, W, n_feat, h):
  """returns (hidden_act, output_pre, output_sigmoid)"""
  Y, z = unpack_weights(W, n_feat, h)

  hidden_pre = X @ Y
  hidden_act = sigmoid(hidden_pre)

  output_pre = hidden_act @ z
  output_sig = sigmoid(output_pre)

  return hidden_act, output_pre, output_sig


def compute_grad(X, y, W, n_feat, h):
    Y, z = unpack_weights(W, n_feat, h)
    #m    = X.shape[0]

    hidden_act, output_pre, y_hat = forward_pass(X, W, n_feat, h)
    residual = y_hat - y

    # chain rule: dL/d(output_pre)  (outer sigmoid part)
    dL_dv = 2.0 * residual * sigmoid_grad(output_pre)

    # --- grads w.r.t z ---
    grad_z = hidden_act.T @ dL_dv

    # --- grads w.r.t Y ---
    factor  = dL_dv[:, None] * z[None, :] * sigmoid_grad(X @ Y)
    grad_Y  = X.T @ factor

    return pack_weights(grad_Y, grad_z)
###############################################################################


########################## Error & Performance. ###############################
def compute_error(X, y, W, n_feat, h):
    _, _, y_hat = forward_pass(X, W, n_feat, h)
    return np.sum((y_hat - y) ** 2)

def accuracy(y_hat, y, doobis_magical_threshold=0.5):
    y_pred = (y_hat >= doobis_magical_threshold).astype(int)
    acc = np.mean(y_pred == y)
    return acc
###############################################################################