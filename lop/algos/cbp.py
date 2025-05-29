from torch import optim
import torch
from lop.algos.gnt import GnT
from lop.utils.AdamGnT import AdamGnT
import torch.nn.functional as F


class ContinualBackprop(object):
    """
    The Continual Backprop algorithm, used in https://arxiv.org/abs/2108.06325v3
    """
    def __init__(
            self,
            net,
            # util_save_dir,
            util_save_every_nth_iteration,
            step_size=0.001,
            loss='mse',
            opt='sgd',
            beta=0.9,
            beta_2=0.999,
            replacement_rate=0.001,
            decay_rate=0.9,
            device='cpu',
            maturity_threshold=100,
            util_type='contribution',
            init='kaiming',
            accumulate=False,
            momentum=0,
            outgoing_random=False,
            weight_decay=0,
            snp_to_perturb=False,
            #snp_shrink_rate=1,
            snp_perturb_scale=0,
    ):
        self.net = net
        self.device = device

        # define the optimizer
        if opt == 'sgd':
            self.opt = optim.SGD(self.net.parameters(), lr=step_size, momentum=momentum, weight_decay=weight_decay)
        elif opt == 'adam':
            self.opt = AdamGnT(self.net.parameters(), lr=step_size, betas=(beta, beta_2), weight_decay=weight_decay)

        # define the loss function
        self.loss_func = {'nll': F.cross_entropy, 'mse': F.mse_loss}[loss]

        # a placeholder
        self.previous_features = None

        # define the generate-and-test object for the given network
        self.gnt = None
        self.gnt = GnT(
            net=self.net.layers,
            hidden_activation=self.net.act_type,
            opt=self.opt,
            # util_save_dir=util_save_dir,
            # util_save_every_nth_iteration=util_save_every_nth_iteration,
            replacement_rate=replacement_rate,
            decay_rate=decay_rate,
            maturity_threshold=maturity_threshold,
            util_type=util_type,
            device=device,
            loss_func=self.loss_func,
            init=init,
            accumulate=accumulate,
        )

        self.util = []
        # self.bias_corrected_util = []
        self.iteration_count = -1
        self.util_save_every_nth_iteration = util_save_every_nth_iteration

        self.snp_to_perturb = snp_to_perturb
        self.snp_perturb_scale = snp_perturb_scale
        #self.snp_shrink_rate = snp_shrink_rate

    def copy_util_score(self, array_of_torch_tensors):
        return [x.clone() for x in array_of_torch_tensors]

    def learn(self, x, target):
        """
        Learn using one step of gradient-descent and generate-&-test
        :param x: input
        :param target: desired output
        :return: loss
        """

        self.iteration_count += 1

        # do a forward pass and get the hidden activations
        output, features = self.net.predict(x=x)
        loss = self.loss_func(output, target)
        self.previous_features = features

        # do the backward pass and take a gradient step
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        if self.snp_to_perturb:
            self.perturb()

        # take a generate-and-test step
        self.opt.zero_grad()
        if type(self.gnt) is GnT:
            self.gnt.gen_and_test(features=self.previous_features)

            if self.iteration_count % self.util_save_every_nth_iteration == 0:
                # Save the utility scores
                cur_util = self.gnt.util  
                cur_bias_corrected_util = self.gnt.bias_corrected_util
                self.util.append(self.copy_util_score(cur_util))
                # self.bias_corrected_util.append(self.copy_util_score(cur_bias_corrected_util))

        if self.loss_func == F.cross_entropy:
            return loss.detach(), output.detach()

        return loss.detach()


    def perturb(self):
        with torch.no_grad():
            for i in range(int(len(self.net.layers)/2)+1):
                # Addition by me: multiply by the shrink rate (as in the original Shrink and Perturb paper)
                #self.net.layers[i * 2].bias *= self.snp_shrink_rate
                #self.net.layers[i * 2].weight *= self.snp_shrink_rate

                # Perturb the weights and biases (already was in the codebase)
                self.net.layers[i * 2].bias +=\
                    torch.empty(self.net.layers[i * 2].bias.shape, device=self.device).normal_(mean=0, std=self.snp_perturb_scale)
                self.net.layers[i * 2].weight +=\
                    torch.empty(self.net.layers[i * 2].weight.shape, device=self.device).normal_(mean=0, std=self.snp_perturb_scale)
