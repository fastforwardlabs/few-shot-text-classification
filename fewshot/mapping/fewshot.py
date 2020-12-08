
import torch 

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

    def train(self, data_loader, num_epochs, lam):
        history = []

        ##### For GPU #######
        if self.device == 'cuda' and torch.cuda.is_available():
            self.cuda()

        for _ in tqdm(range(num_epochs), desc="Epoch"):
            total_loss = 0   
            for step, batch in enumerate(data_loader):
                batch = tuple(t.to(self.device) for t in batch)
                X_batch = batch[0]
                Y_batch = batch[1]

                output = model(X_batch)

                loss = self.loss_fcn(output, Y_batch, self.linear.weight, lam) 
                total_loss += loss.item()

                self.optimizer.zero_grad()                
                loss.backward()
                self.optimizer.step()

            history.append(total_loss)
      
        return history


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
        # TODO: think through how to explain this!!
        identity = torch.eye(w.size()[1], device=self.device)
        err2 = torch.sum((w - identity)**2) / x.data.nelement()
        return err1 + lam*err2

