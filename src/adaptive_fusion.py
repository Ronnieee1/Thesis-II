import numpy as np

def adaptive_control(alpha, beta, val_loss_a, val_loss_b, gamma=0.1):
    # Exponential moving average for validation loss
    exp_a = np.exp(-gamma * val_loss_a)
    exp_b = np.exp(-gamma * val_loss_b)
    alpha_t = exp_a / (exp_a + exp_b)
    beta_t = 1 - alpha_t
    return alpha_t, beta_t

def final_prediction(alpha, beta, p_a, p_b):
    return alpha * p_a + beta * p_b
