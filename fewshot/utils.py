import os
import pickle
import torch
import torch.nn.functional as F


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def to_tensor(vector):
    # Something is wrong with this and I have no idea what
    tensor = torch.tensor(vector, dtype=torch.float)
    return tensor
    # return torch.Tensor(vector, dtype=torch.float)


def save_as_tensor(vector, filename, overwrite=False):
    if os.path.exists(filename) and not overwrite:
        print(f"{filename} already exists! Please use overwrite flag.")
    else:
        torch.save(torch.tensor(vector, dtype=torch.float), filename)


def load_tensor(filename, to_gpu=False):
    if os.path.exists(filename):
        if to_gpu:
            return torch.load(filename)
        return torch.load(filename, map_location=torch.device("cpu"))
    else:
        print(f"{filename} does not exist!")


def save_as_vector(vector, filename, overwrite=False):
    if os.path.exists(filename) and not overwrite:
        print(f"{filename} already exists! Please use overwrite flag.")
    else:
        pickle.dump(vector, open(filename, "wb"))


def load_vector(filename):
    if os.path.exists(filename):
        return pickle.load(open(filename, "rb"))
    else:
        print(f"{filename} does not exist!")


def check_path(pathname):
    newdir = "/".join(pathname.split("/")[:-1])
    if not os.path.exists(newdir):
        os.makedirs(newdir)


def compute_projection_matrix(X, Y):
    """
  compute projection matrix of best fit, w, that transforms X to Y according to:

  Y = Xw

  (X.T X)^-1 X.T Y = [(X.T X)^-1 X.T X]w

  w = (X.T X)^-1 X.T Y
  """
    X_norm = F.normalize(X, p=2, dim=1)
    Y_norm = F.normalize(Y, p=2, dim=1)

    Z = torch.inverse(torch.matmul(X_norm.T, X_norm))
    Z = torch.matmul(Z, X_norm.T)
    w = torch.matmul(Z, Y_norm)

    return w
