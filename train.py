import torch


class Trainer(object):
    def __init__(self, dataloader, config) -> None:
        self.dataloader = dataloader

    def train(self) -> None:
        data_iter = iter(self.dataloader)
