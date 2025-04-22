###############################################################################
## Description: Defines the utility functions performing newton update using ##
##              the BFGS/SR1 update algorithm discussed in the oringal paper ##                                     
###############################################################################

import torch
from torch import Tensor
from torch import device as torch_device
from Utils.neural_network import unpack_weights, compute_error, compute_grad
from tqdm import trange
from typing import Tuple, List


############################# Computing Deltas ################################
def compute_gradient_delta(X, y, W_prev, W_current, n_feat, h):
    """
        Description: Computes difference in gradients
    """
    gamma_prev = compute_grad(X, y, W_prev, n_feat, h)
    gamma_curr = compute_grad(X, y, W_current, n_feat, h)
    delta_gamma = gamma_curr - gamma_prev
    return delta_gamma

def compute_weight_delta(W_prev, W_current):
    """
        Description: Computs differences in weights
    """
    weight_delta = W_current - W_prev
    return weight_delta
###############################################################################


############################## Hessian Update #################################
def should_use_BFGS_update(H_inv, grad_delta, weight_delta, eps=1e-8) -> bool:
    """ True if use BFGS; False otherwise"""
    Hy = H_inv @ grad_delta
    denom = torch.dot(weight_delta - Hy, grad_delta)
    return denom.item() <= eps

def compute_BFGS_Hessian_update(Hessian_prev, grad_delta, weight_delta, device,
                                stop_threshold = 1e-12):
    """ Compute Inverse Hessian Approximate with BFGS method """
    _outer = lambda a, b: a.unsqueeze(1) @ b.unsqueeze(0)   # (p,1)(1,p)->(p,p)
    s, y = weight_delta, grad_delta
    # To prevent  gradient from going to NaN, stop updating the Hessian when the 
    # denominator used in its calculation becomes smaller than the threshold
    denominator = y.T @ s
    if denominator.abs() < stop_threshold:
        return Hessian_prev
    rho = 1.0 / denominator
    I    = torch.eye(len(Hessian_prev), dtype=Hessian_prev.dtype, device = device)
    V    = I - rho * _outer(s, y)
    return V @ Hessian_prev @ V.T + rho * _outer(s, s)

def compute_SR1_Hessian_update(Hessian_prev, grad_delta, weight_delta):
    """ Compute Inverse Hessian Approximate with SR1 method """
    _outer = lambda a, b: a.unsqueeze(1) @ b.unsqueeze(0)   # (p,1)(1,p)->(p,p)
    s, y = weight_delta, grad_delta
    Hy   = Hessian_prev @ y
    diff = s - Hy
    denom= diff.T @ y
    
    return Hessian_prev + _outer(diff, diff) / denom

def update_inverse_hessian(H_prev, grad_delta, weight_delta, device, eps=1e-8):
    if should_use_BFGS_update(H_prev, grad_delta, weight_delta, eps):
        return compute_BFGS_Hessian_update(H_prev, grad_delta, weight_delta, device)
    return compute_SR1_Hessian_update(H_prev, grad_delta, weight_delta)
###############################################################################


################# Lr Update and Weight Update & Minimization ##################
def perform_line_search(f, grad_f, W, d_k):
	lambda_ = 1.0
	max_iter = 30
	fw = f(W)
	grad_fw = grad_f(W)
	grad_fw_d = torch.dot(grad_fw, d_k)

	for i in range(max_iter):
		W_lamb_d = W + lambda_ * d_k
		fw_lamb_d = f(W_lamb_d)
		grad_fw_lamb_d = grad_f(W_lamb_d)
		# 1st condition
		if fw_lamb_d - fw > (1e-4) * lambda_ * grad_fw_d:
			lambda_ *= 0.75
		# 2nd condition

		elif torch.dot(grad_fw_lamb_d, d_k) < 0.9 * grad_fw_d:
			lambda_ *= 1.2
		# have met wolfe conditions
		else:
			return lambda_

	return lambda_


def update_weights(X, y_true, W, H_inv, n_feat, h):
    f       = lambda w: compute_error(X, y_true, w, n_feat, h)
    grad_f  = lambda w: compute_grad (X, y_true, w, n_feat, h)

    g_W     = grad_f(W)
    d_k     = - H_inv @ g_W

    alpha_k = perform_line_search(f, grad_f, W, d_k)

    W_new   = W + alpha_k * d_k

    return W_new

def grad_threshold_reached(grad_norm: float, W: Tensor, grad_tol: float) -> bool:
    """"
        Returns true if the norm of the gradients has reached the 
        stopping threshold
    """
    if grad_norm <= grad_tol * max(1.0, torch.norm(W).item()):
        return True
    else:
        return False
    
def compute_minimizer(W_init: Tensor, X: Tensor, y_true: Tensor,
                      n_hidden: int, device: torch_device, max_iters: int = 1000,
                      grad_tol: float = 1e-8) -> Tuple[Tensor, List]:
    """ 
        Description: Computes the approximate minimizer W* for the given
                        neural network on the given problem using the
                        quasi-Newton method with SR1/BFGS update algorithm
        Args:
            W_init (Tensor): Initial weights of Neural Network
            n_hidden (int): Number of units in hidden layer
            device (torch_device): Device on which to perform the 
                                    computation/minimization
            max_iters (int): Maximum number of iterations to run
            grad_tol (float): fraction of the gradient indicating when to
                            stop minimization
        Returns
            W_optimal (Tensor): Optimzed set of weights for given Neural Network
            training_losses (list): List of training losses/errors during training
    """
    # Initialize weights and inverse Hessian
    W = W_init
    H_inv = torch.eye(W.shape[0], device=device) # Intialize inverse hessian as Identity matrix 
                                # (satisfying PSD condition)
    n_feats = X.shape[1]

    # Track the weights with lowest error
    loss_fn       = lambda w: compute_error(X, y_true, w, n_feats, n_hidden)
    training_losses = []
    best_loss = torch.inf
    W_best = W.clone()
    progress_bar = trange(max_iters, desc="Optimizing", leave=True)
    for i in progress_bar:
        current_loss = loss_fn(W).item()
        training_losses.append(current_loss)
        if current_loss < best_loss:
             best_loss = current_loss
             W_best = W.clone()
        # Break if gradients are sufficiently small
        gradients = compute_grad(X, y_true, W, n_feats, n_hidden)
        grad_norm = torch.norm(gradients).item()
        progress_bar.set_description(f"Iteration {i} | grad norm: {grad_norm:.2e}")
        if grad_threshold_reached(grad_norm, W, grad_tol):
            break
        else:
            # Otherwise continue updating weights and inverse hessian
            W_new = update_weights(X, y_true, W, H_inv, n_feats, n_hidden)

            grad_delta = compute_gradient_delta(X, y_true, W_prev=W, W_current=W_new,
                                                n_feat = n_feats, h=n_hidden)
            weight_delta = compute_weight_delta(W_prev=W, W_current=W_new)
            H_inv = update_inverse_hessian(H_inv, grad_delta, weight_delta, device)
            W = W_new

    W_optimal = W_best
    return W_optimal, training_losses
###############################################################################
