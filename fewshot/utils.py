# ###########################################################################
#
#  CLOUDERA APPLIED MACHINE LEARNING PROTOTYPE (AMP)
#  (C) Cloudera, Inc. 2021
#  All rights reserved.
#
#  Applicable Open Source License: Apache 2.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# ###########################################################################

import os
import pickle
import pathlib

import torch


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
        with open(filename, "rb") as f:
            return pickle.load(f)
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
