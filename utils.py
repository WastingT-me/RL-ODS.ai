import torch.nn as nn
import torch.optim as optim
import numpy as np

class Definitions:
    @staticmethod
    def get_optimizer(optimizer_name, model_parameters, lr):
        if optimizer_name == 'adam':
            return optim.Adam(model_parameters, lr=lr)
        elif optimizer_name == 'sgd':
            return optim.SGD(model_parameters, lr=lr)
        elif optimizer_name == 'rmsprop':
            return optim.RMSprop(model_parameters, lr=lr)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not recognized.")

    @staticmethod
    def get_scheduler(scheduler_name, optimizer, **kwargs):
        if scheduler_name == 'step_lr':
            return optim.lr_scheduler.StepLR(optimizer, **kwargs)
        elif scheduler_name == 'exponential_lr':
            return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
        elif scheduler_name == 'cosine_annealing_lr':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
        else:
            raise ValueError(f"Scheduler {scheduler_name} not recognized.")

    @staticmethod
    def get_loss_function(loss_name):
        if loss_name == 'mse':
            return nn.MSELoss()
        elif loss_name == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif loss_name == 'huber':
            return nn.SmoothL1Loss()
        else:
            raise ValueError(f"Loss function {loss_name} not recognized.")

# Simple Environment Interface
class SimpleEnv:
    def __init__(self):
        self.state_dim = 4
        self.action_n = 2

    def reset(self):
        return np.random.randn(self.state_dim)

    def step(self, action):
        next_state = np.random.randn(self.state_dim)
        reward = np.random.randn()
        done = np.random.rand() > 0.95
        return next_state, reward, done, {}