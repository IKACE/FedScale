import logging
import math
import os
import sys

import numpy as np
import torch
from clip_norm import clip_grad_norm_
from privacy_accountant import Privacy_Accountant
from torch.autograd import Variable

from fedscale.cloud.execution.client import Client


class Customized_Client(Client):
    """
    Basic client component in Federated Learning
    Local differential privacy
    With a privacy accountant
    """
    def __init__(self, conf):
        self.noise_factor = conf.noise_factor
        self.clip_threshold = conf.clip_threshold
        self.privacy_acct = Privacy_Accountant({
            "eps_budget": 10,
            "delta": 1e-6,
        })
        
        super().__init__(conf)

    def train(self, client_data, model, conf):

        clientId = conf.clientId
        logging.info(f"Start to train (CLIENT: {clientId}) ...")
        tokenizer, device = conf.tokenizer, conf.device
        last_model_params = [p.data.clone() for p in model.parameters()]

        model = model.to(device=device)
        model.train()

        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size)
        self.global_model = None

        if conf.gradient_policy == 'fed-prox':
            # could be move to optimizer
            self.global_model = [param.data.clone() for param in model.parameters()]

        optimizer = self.get_optimizer(model, conf)
        criterion = self.get_criterion(conf)
        error_type = None

        # TODO: One may hope to run fixed number of epochs, instead of iterations
     
        while self.completed_steps < conf.local_steps:
            
            try:
                self.train_step(client_data, conf, model, optimizer, criterion)
            except Exception as ex:
                error_type = ex
                break

        delta_weight = []
        for param in model.parameters():
            delta_weight.append((param.data.cpu() - last_model_params[len(delta_weight)]))

        clip_grad_norm_(delta_weight, max_norm=conf.clip_threshold)

        # recover model weights
        idx = 0
        for param in model.parameters():
            param.data = last_model_params[idx] + delta_weight[idx]
            idx += 1
        sigma = conf.noise_factor * conf.clip_threshold
        state_dicts = model.state_dict()
        model_param = {p:  np.asarray(state_dicts[p].data.cpu().numpy() + \
            torch.normal(mean=0, std=sigma, size=state_dicts[p].data.shape).cpu().numpy()) for p in state_dicts}


        results = {'clientId': clientId, 'moving_loss': self.epoch_train_loss,
                   'trained_size': self.completed_steps*conf.batch_size, 'success': self.completed_steps > 0}
        results['utility'] = math.sqrt(
            self.loss_squre)*float(trained_unique_samples)
        results['sigma'] = conf.noise_factor
        results['sample_rate'] = conf.batch_size / len(client_data.dataset)
        results['niter'] = conf.local_steps # caution on this

        if error_type is None:
            logging.info(f"Training of (CLIENT: {clientId}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {clientId}) failed as {error_type}")

        results['update_weight'] = model_param
        results['wall_duration'] = 0

        return results




