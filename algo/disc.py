import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from .utils import soft_update, hard_update
from .model import Predictor, MLP, Discriminator

class PredTrainer():
    def __init__(self, obs_shape, args) -> None:
        self.args = args
        self.obs_shape = obs_shape

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.pred = Predictor(obs_shape, args.num_modes, 256, obs_shape, output_std=1.0).to(device=self.device)
        self.pred_optim = Adam(self.pred.parameters(), lr=args.pred_lr)

    def score(self, state, label, state_delta):
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        if len(label.shape) == 1:
            label = np.expand_dims(label, axis=0)
        if len(state_delta.shape) == 1:
            state_delta = np.expand_dims(state_delta, axis=0)
        state = torch.FloatTensor(state).to(self.device)
        label = torch.FloatTensor(label).to(self.device)
        state_delta = torch.FloatTensor(state_delta).to(self.device)
        self.pred.eval()
        log_prob, _ = self.pred.evaluate(state, label, state_delta)
        self.pred.train()
        return log_prob.squeeze().detach().cpu().numpy()

    def update_parameters(self, samples):
        state_batch, label_batch, state_delta_batch = samples
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        label_batch = torch.FloatTensor(label_batch).to(self.device)
        state_delta_batch = torch.FloatTensor(state_delta_batch).to(self.device)
        log_prob, _ = self.pred.evaluate(state_batch, label_batch, state_delta_batch)
        loss = -torch.mean(log_prob)

        self.pred_optim.zero_grad()
        loss.backward()
        self.pred_optim.step()

        return loss.item()

class DiscTrainer(object):
    def __init__(self, obs_shape, args):
        self.args = args
        self.obs_shape = obs_shape

        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.disc = Discriminator(obs_shape, args.hidden_dim, args.num_modes).to(device=self.device)
        self.disc_optim = Adam(self.disc.parameters(), lr=args.disc_lr)

    def score(self, state, labels):
        if len(state.shape) == 1:
            state = np.expand_dims(state, axis=0)
        if len(labels.shape) == 1:
            labels = np.expand_dims(labels, axis=0)
        state = torch.FloatTensor(state).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)
        _, lpgts, _ = self.disc(state, labels)
        return lpgts.squeeze().detach().cpu().numpy()

    def update_parameters(self, samples):
        context_batch, state_batch = samples
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        context_batch = torch.FloatTensor(context_batch).to(self.device)
        _, lpgts, _ = self.disc(state_batch, context_batch)

        loss = -torch.mean(lpgts)

        self.disc_optim.zero_grad()
        loss.backward()
        self.disc_optim.step()

        return loss.item()