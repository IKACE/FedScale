# -*- coding: utf-8 -*-

import os
import sys

from privacy_accountant import Privacy_Accountant


from fedscale.cloud.aggregation.aggregator import Aggregator
import fedscale.cloud.config_parser as parser

"""In this example, we only need to change the Client Component we need to import"""

class Customized_Aggregator(Aggregator):
    """Each executor takes certain resource to run real training.
       Each run simulates the execution of an individual client"""

    def __init__(self, args):
        super(Customized_Aggregator, self).__init__(args)


if __name__ == "__main__":
    aggregator = Customized_Aggregator(parser.args)
    aggregator.run()

