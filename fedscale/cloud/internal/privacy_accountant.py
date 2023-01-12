import logging
import wandb
from autodp.autodp_core import Mechanism
from autodp.transformer_zoo import Composition 
from autodp import mechanism_zoo, transformer_zoo

class Privacy_Accountant(object):
    """ A privacy accountant for recording privacy loss and calculating remaining privacy budget
    """
    def __init__(self, args) -> None:
        self.eps_budget = args['eps_budget']
        self.delta = args['delta']
        self.clientId = args['clientId']
        self.eps_loss = 0
        self.changed = False
        self.niter = 0
        # record each mechanism and the number of times being used
        # each element should be {"mech": the actual mechanism used,
        #                         "sigma": Gaussian noise scale sigma,
        #                         "sample_rate": subsampling rate,
        #                         "niter": number of iterations this mech is used}
        self.context = []

    def get_privacy_loss(self):
        if self.changed:
            self.compute_budget()
        return self.eps_loss
    
    def get_remaining_budget(self):
        if self.changed:
            self.compute_budget()
        return max(self.eps_budget-self.eps_loss, 0)
    
    def get_total_budget(self):
        return self.eps_budget

    def update_budget(self, params):
        """Update privacy budget by updating the mechanisms used in context, do not compute privacy budget here for efficiency
        """
        self.niter += params["niter"]
        if len(self.context) == 0:
            subsample = transformer_zoo.AmplificationBySampling() 
            mech = mechanism_zoo.GaussianMechanism(sigma=params["sigma"])
            # Create subsampled Gaussian mechanism
            mech = subsample(mech,params["sample_rate"],improved_bound_flag=True)
            self.context.append({
                "mech": mech,
                "sigma": params["sigma"],
                "sample_rate": params["sample_rate"],
                "niter": params["niter"],
            })

        for mech_info in self.context:
            # if this is an already known mechanism
            if abs(params["sigma"]-mech_info["sigma"]) < 0.00001 and abs(params["sample_rate"]-mech_info["sample_rate"]) < 0.00001:
                mech_info["niter"] += params["niter"]
                logging.info(f"Updating mechanism niter: {mech_info['niter']}")
            # if this is a new mechanism
            else:
                subsample = transformer_zoo.AmplificationBySampling() 
                mech = mechanism_zoo.GaussianMechanism(sigma=params["sigma"])
                # Create subsampled Gaussian mechanism
                mech = subsample(mech,params["sample_rate"],improved_bound_flag=True)
                self.context.append({
                    "mech": mech,
                    "sigma": params["sigma"],
                    "sample_rate": params["sample_rate"],
                    "niter": params["niter"],
                })

        self.changed = True

    def compute_budget(self):
        if self.changed == False:
            return
        compose = transformer_zoo.Composition()
        composed_mech = compose([mech_info["mech"] for mech_info in self.context], [mech_info["niter"] for mech_info in self.context])
        # self.print()
        self.eps_loss = composed_mech.get_approxDP(self.delta)
        self.print()
        self.changed = False

    def print(self):
        # logging.info(f"=== Client {self.clientId} Privacy Accounting Info Begins ===")
        # logging.info(f'Total Epsilon Budget: {self.eps_budget}')
        logging.info(f'Client {self.clientId} Current Epsilon Loss: {self.eps_loss}/{self.eps_budget}')
        # this is ad-hoc
        if self.clientId == 1:
            wandb.log({"Client/privacy_loss": self.eps_loss,
                       "niter": self.niter})
        # logging.info(f"=== Client {self.clientId} Privacy Accounting Info Ends ===")