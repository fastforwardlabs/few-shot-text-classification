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

# from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.notebook import tqdm


class FewShotLinearRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, loss_fcn, lr, device=None):
        super(FewShotLinearRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=False)

        # initialize the weights in the linear layer to zeros
        torch.nn.init.zeros_(self.linear.weight)

        self.loss_fcn = loss_fcn
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        self.device = device

    def forward(self, x):
        out = self.linear(x)
        return out


class BayesianMSELoss(torch.nn.Module):
    """ Mean Reciprocal Rank Loss """

    def __init__(self, device=None):
        super(BayesianMSELoss, self).__init__()
        self.device = device

    def forward(self, x, y, w, lam):
        # The first part of the loss function is just standard MSE
        # and represents our prior ??
        err1 = torch.nn.functional.mse_loss(x, y)
        # The second part ...
        # TODO(#26): think through how to explain this!!
        identity = torch.eye(w.size()[1], device=self.device)
        err2 = torch.sum((w - identity) ** 2) / x.data.nelement()
        return err1 + lam * err2


def prepare_dataloader(dataset, Zmap=None, batch_size=50):
    """Convert a Dataset object to a PyTorch DataLoader object for
    training Wmap

    Include Zmap if Wmap should be trained on SBERT*Zmap representations
    """
    example_embeddings = dataset.embeddings[: -len(dataset.categories)]
    if Zmap is not None:
        X_train = torch.mm(example_embeddings, Zmap)
        y_train = torch.mm(dataset.label_embeddings, Zmap)
    else:
        X_train = example_embeddings
        y_train = dataset.label_embeddings

    tensor_dataset = TensorDataset(X_train, y_train)
    data_loader = DataLoader(tensor_dataset, shuffle=True, batch_size=batch_size)

    return data_loader


def train(model, data_loader, num_epochs, lam, device="cuda"):
    history = []

    ##### Double check GPUs are available #######
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    model.to(device)

    # for _ in tqdm(range(num_epochs), desc="Epoch"):
    for _ in range(num_epochs):
        total_loss = 0
        for step, batch in enumerate(data_loader):
            batch = tuple(t.to(device) for t in batch)
            X_batch = batch[0]
            Y_batch = batch[1]

            output = model(X_batch)

            loss = model.loss_fcn(output, Y_batch, model.linear.weight, lam)
            total_loss += loss.item()

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

        history.append(total_loss)

    return history
