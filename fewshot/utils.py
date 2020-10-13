import os
import pickle
import torch

def to_list(tensor):
    return tensor.detach().cpu().tolist()

def to_tensor(list):
    return torch.tensor(list, dtype=tensor.float)

def save_as_tensor(vector, filename, overwrite=False):
    if os.path.exists(filename) and not overwrite:
        print(f"{filename} already exists! Please use overwrite flag.")
    else:
        torch.save(torch.tensor(vector, dtype=torch.float), filename)

def load_tensor(filename, to_gpu=False):
    if os.path.exists(filename):
        if to_gpu:
            return torch.load(filename)
        return torch.load(filename, map_location=torch.device('cpu'))
    else:
        print(f"{filename} does not exist!")

def save_as_vector(vector, filename, overwite=False):
    if os.path.exists(filename) and not overwrite:
        print(f"{filename} already exists! Please use overwrite flag.")
    else:
        pickle.dump(vector, open(filename), 'wb')

def load_vector(filename):
    if os.path.exists(filename):
        return pickle.load(open(filename), "rb")
    else:
        print(f"{filename} does not exist!")    

def compute_projection_matrix(X, Y):
  """
  compute projection matrix of best fit, w, that transforms X to Y according to:

  Y = Xw

  (X.T X)^-1 X.T Y = [(X.T X)^-1 X.T X]w

  w = (X.T X)^-1 X.T Y
  """

  Z = torch.inverse(torch.matmul(X.T, X))
  Z = torch.matmul(Z, X.T)
  w = torch.matmul(Z, Y)

  return w