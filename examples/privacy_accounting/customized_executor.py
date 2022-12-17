# -*- coding: utf-8 -*-

import os
import sys

from privacy_accountant import Privacy_Accountant
from customized_client import Customized_Client

from fedscale.cloud.execution.executor import Executor
import fedscale.cloud.config_parser as parser

"""In this example, we only need to change the Client Component we need to import"""

class Customized_Executor(Executor):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super(Customized_Executor, self).__init__(args)
        # mapping from client_id to client's privacy accountant
        self.context = {}

    # def training_handler(self, clientId, conf, model=None):
    #     """Train model given client id, record privacy loss information
        
    #     Args:
    #         clientId (int): The client id.
    #         conf (dictionary): The client runtime config.

    #     Returns:
    #         dictionary: The train result
        
    #     """
    #     train_res = super().training_handler(self, clientId, conf, model=None)
    #     if clientId not in self.context:
    #         self.context[clientId] = Privacy_Accountant({
    #             "eps_budget": 10,
    #             "delta": 1e-6,
    #         })
    #     client_acct = self.context[clientId]
    #     return train_res

    def get_client_trainer(self, conf):
        return Customized_Client(conf)

if __name__ == "__main__":
    executor = Customized_Executor(parser.args)
    executor.run()

