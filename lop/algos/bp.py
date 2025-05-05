import torch
import torch.nn.functional as F
from lop.algos.gnt import GnT
from torch import optim


class Backprop(object):
    def __init__(self, net, step_size=0.001, loss='mse', opt='sgd', beta_1=0.9, beta_2=0.999, weight_decay=0.0,
                 to_perturb=False, perturb_scale=0.1, device='cpu', momentum=0,
                 decay_rate=0.99 # default value for decay rate (as in the config files for Continual Backprop)
                 ):
        self.net = net
        self.to_perturb = to_perturb
        self.perturb_scale = perturb_scale
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, weight_decay=weight_decay, momentum=momentum)
        elif opt == 'adam':
            self.opt = optim.Adam(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                  weight_decay=weight_decay)
        elif opt == 'adamW':
            self.opt = optim.AdamW(self.net.parameters(), lr=step_size, betas=(beta_1, beta_2),
                                   weight_decay=weight_decay)

        # define the loss function
        self.loss = loss
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[self.loss]

        # Placeholder
        self.previous_features = None

        # define the generate-and-test object for the given network
        # Used for logging the utility scores
        self.gnt = None
        self.gnt = GnT(
            net=self.net.layers,
            hidden_activation=self.net.act_type,
            opt=self.opt,
            decay_rate=decay_rate,
            util_type="adaptable_contribution", # same as in the config files for Continual Backprop
            device=device,
            loss_func=self.loss_func,
        )

        self.util = []
        self.bias_corrected_util = []

    def copy_util_score(self, array_of_torch_tensors):
        return [x.clone() for x in array_of_torch_tensors]

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent
        :param x: input
        :param target: desired output
        :return: loss
        """
        self.opt.zero_grad()
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        loss.backward()
        self.opt.step()
        if self.to_perturb:
            self.perturb()

        if type(self.gnt) is GnT:
            self.gnt.update_utility_for_logging(features=self.previous_features)
            cur_util = self.gnt.util
            cur_bias_corrected_util = self.gnt.bias_corrected_util
            self.util.append(self.copy_util_score(cur_util))
            self.bias_corrected_util.append(self.copy_util_score(cur_bias_corrected_util))

        if self.loss == 'nll':
            return loss.detach(), output.detach()
        return loss.detach()

    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.perturb_scale)
