import os
import pickle
import pathlib

import torch
import torch.nn.functional as F


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def to_tensor(vector):
    # Something is wrong with this and I have no idea what
    tensor = torch.tensor(vector, dtype=torch.float)
    return tensor
    # return torch.Tensor(vector, dtype=torch.float)


def torch_save(vector, filename, overwrite=False):
    if os.path.exists(filename) and not overwrite:
        print(f"{filename} already exists! Please use overwrite flag.")
    else:
        create_path(filename)
        torch.save(torch.tensor(vector, dtype=torch.float), filename)


def torch_load(filename, to_gpu=False):
    if os.path.exists(filename):
        if to_gpu:
            return torch.load(filename)
        return torch.load(filename, map_location=torch.device("cpu"))
    else:
        print(f"{filename} does not exist!")


def pickle_save(vector, filename, overwrite=False):
    if os.path.exists(filename) and not overwrite:
        print(f"{filename} already exists! Please use overwrite flag.")
    else:
        create_path(filename)
        pickle.dump(vector, open(filename, "wb"))


def pickle_load(filename):
    if os.path.exists(filename):
        return pickle.load(open(filename, "rb"))
    else:
        print(f"{filename} does not exist!")


def create_path(pathname: str) -> None:
    """Creates the directory for the given path if it doesn't already exist."""
    dir = str(pathlib.Path(pathname).parent)
    if not os.path.exists(dir):
        os.makedirs(dir)


def fewshot_filename(*paths) -> str:
    """Given a path relative to this project's top-level directory, returns the
    full path in the OS.

    Args:
        paths: A list of folders/files.  These will be joined in order with "/"
            or "\" depending on platform.

    Returns:
        The full absolute path in the OS.
    """
    # First parent gets the scripts directory, and the second gets the top-level.
    result_path = pathlib.Path(__file__).resolve().parent.parent
    for path in paths:
        result_path /= path
    return str(result_path)
