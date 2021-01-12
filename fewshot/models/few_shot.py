from tqdm.notebook import tqdm, tnrange

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler


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
    """ Convert a Dataset object to a PyTorch DataLoader object for 
        training Wmap

        Include Zmap if Wmap should be trained on SBERT*Zmap representations
    """
    example_embeddings = dataset.embeddings[:-len(dataset.categories)]
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

    ##### For GPU #######
    if device == "cuda" and torch.cuda.is_available():
        model.to("cuda")

    for _ in tqdm(range(num_epochs), desc="Epoch"):
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
