import torch.nn as nn
import torch.optim as optim


def get_criterion():
    return nn.CrossEntropyLoss()


def get_optimizer(net, learning_rate, momentum):
    return optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)