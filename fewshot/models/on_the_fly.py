import torch
import torch.nn.functional as F 


def OLS_with_l2_regularization(X: torch.Tensor, Y: torch.Tensor, alpha: float = 0) -> torch.Tensor:
    """Computes ordinary least squares

    Ordinary least squares with l2 regularization can be expressed in closed form, 
    meaning that we do not need to perform gradient descent in order
    to find the best fit solution! 
    For more information on the derivation of the closed-form expression, 
    check it the Wikipedia page here: 
    https://en.wikipedia.org/wiki/Ordinary_least_squares#Matrix/vector_formulation

    In brief: we find a matrix, w, that transforms X to Y according to:

    Y = Xw

    (X.T X)^-1 X.T Y = [(X.T X)^-1 X.T X]w

    w = (X.T X + alpha*I)^-1 X.T Y

    where I is the identity matrix and alpha is the amount of regularization. 
    alpha = 0 is equivalent to OLS (ordinary least squares)
    alpha >= 0 is ridge regression / l2 regularization
    """
    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)
    I = torch.eye(X_norm.shape[1])

    inner = torch.matmul(X_norm.T, X_norm) + alpha*I
    Z = torch.inverse(inner)
    Z = torch.matmul(Z, X_norm.T)
    w = torch.matmul(Z, Y_norm)

    return w