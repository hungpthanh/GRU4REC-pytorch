import torch.optim as optim


class Optimizer:
    def __init__(self, params, optimizer_type='Adagrad', lr=.05,
                 momentum=0, weight_decay=0, eps=1e-6):
        '''
        An abstract optimizer class for handling various kinds of optimizers.
        You can specify the optimizer type and related parameters as you want.
        Usage is exactly the same as an instance of torch.optim

        Args:
            params: torch.nn.Parameter. The NN parameters to optimize
            optimizer_type: type of the optimizer to use
            lr: learning rate
            momentum: momentum, if needed
            weight_decay: weight decay, if needed. Equivalent to L2 regulariztion.
            eps: eps parameter, if needed.
        '''
        if optimizer_type == 'RMSProp':
            self.optimizer = optim.RMSprop(params, lr=lr, eps=eps, weight_decay=weight_decay, momentum=momentum)
        elif optimizer_type == 'Adagrad':
            self.optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_type == 'Adadelta':
            self.optimizer = optim.Adadelta(params, lr=lr, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == 'Adam':
            self.optimizer = optim.Adam(params, lr=lr, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == 'SparseAdam':
            self.optimizer = optim.SparseAdam(params, lr=lr, eps=eps)
        elif optimizer_type == 'SGD':
            self.optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()