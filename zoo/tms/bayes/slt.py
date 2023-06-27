import itertools
from copy import deepcopy

import numpy as np
import torch


class LearningMachine(object):
    """
    Our model is p(y|x,w) where x, y \in R^n and sigma = 1/sqrt(truth_gamma)
    
          C = sigma * (2pi)^{n/2}
          p(y|x,w) = 1/C exp(-1/(2sigma^2)|| y - ReLU(W^TWx + b) ||^2)
          p(x,y|w) = p(y|x,w)p(x)
    
    Hence the empirical negative log loss of a sample {x_1, ... , x_k} is (n being args.n)
    
          L_k(w) = -1/k \sum_{i=1}^k log p(y_i,x_i|w)
                 = -1/k \sum_{i=1}^k ( logp(y_i|x_i,w) + logp(x_i) )
                 = logC - 1/k \sum_{i=1}^k logp(y_i|x_i,w) - 1/k \sum_{i=1}^k logp(x_i)
                 = logC + L'_k(w) + S_n
    
    Where L'_k(w) = -1/k \sum_{i=1}^k logp(y_i|x_i,w) and S_n is empirical entropy. That is, 
    
          L'_k(w) = truth_gamma/2 1/k \sum_{i=1}^k || y_i - ReLU(W^TWx_i + b ) ||^2
    
    Note that in computing the difference between log losses, the other terms cancel
    out and can be ignored
    
     E_w^\beta[ NL_N(w) ] - NL_N(w_0) = E_w^\beta[ NL'_N ] - NL'_N(w_0)
    
    Hence in the following we use L' instead of L.

    NOTE: we can try scaling the number of SGLD steps with logn

    NOTE: There is also a problem that as N grows, but M remains fixed, the gap may be bigger

    Per dataset variability is unexamined so far
    """
    def __init__(self, net, trainloader, criterion, optimizer, device, truth_gamma, sgld_chains):
        self.net = net
        self.trainloader = trainloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.truth_gamma = truth_gamma
        self.sgld_chains = sgld_chains

        self.batch_size = trainloader.batch_size
        self.total_train = len(self.trainloader.dataset) 

    # Computes NL'_N where N is the training set size
    def compute_energy(self,dataset=None):
        dataset = dataset or self.trainloader
        energies = []
        with torch.no_grad():
            for data in dataset:
                x = data[0].to(self.device)
                outputs = self.net(x)
                batch_size = x.size(0)
                batchloss = self.criterion(outputs, x) * batch_size

                energies.append(batchloss)

        total_energy = sum(energies) * self.truth_gamma / 2
        return total_energy

    # Computes N E_w^\beta[ L'_M ] with L'_M as a surrogate for L'_N and 
    # with SGLD performing the posterior integral
    def compute_local_free_energy(self, num_batches=20, dataset=None):
        dataset = dataset or self.trainloader
        dataiter = iter(dataset)
        first_items = list(itertools.islice(dataiter, num_batches))
        
        # Computes L'_M where M = num_batches * args.batch_size
        def closure():
            total = 0
            for data in first_items:
                x = data[0].to(self.device)
                outputs = self.net(x)
                batchloss = self.criterion(outputs, x)
                total += batchloss

            total = (total * self.truth_gamma) / (2 * num_batches)
            self.optimizer.zero_grad()
            total.backward()

            return total

        # Store the current parameter
        param_groups = []
        for group in self.optimizer.param_groups:
            group_copy = dict()
            group_copy["params"] = deepcopy(group["params"])
            param_groups.append(group_copy)
        
        num_iter = 100 # int(100 * np.log(total_train)/np.log(2000))
        gamma = 1000
        epsilon = 0.1 / 10000
        Lms = []

        def reset_chain():
            # Reset the optimiser's parameter to what it was before SGLD
            for group_index, group in enumerate(self.optimizer.param_groups):
                for param_index, p in enumerate(group["params"]):
                    w = param_groups[group_index]["params"][param_index]
                    p.data.copy_(w.data)

        # SGLD is from M. Welling, Y. W. Teh "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
        # We use equation (4) there, which says given n samples x_1, ... , x_N (here w = (W,b))
        #
        #  w' - w = epsilon_t / 2 ( \grad logp(w) - N \grad L'_N(w) ) + eta_t
        #
        # where eta_t is Gaussian noise, sampled from N(0, \epsilon_t), and p(w) is a prior. We take
        # this to be Gaussian centered at some fixed w_0 parameter, with covariance matrix 1/gamma I_d.
        #
        #   p(w) proportional to exp(-1/2|| w - w_0 ||^2 * gamma)
        #   \grad logp(w) = \grad( -1/2|| w - w_0 ||^2 * gamma ) = -gamma( w - w_0 ).
        #
        # We use a tempered posterior, which means replacing L'_N by \beta L'_N, at inverse temperature
        # \beta = 1/logN.

        for _ in range(self.sgld_chains):
            # This outer loop is over SGLD chains, starting at the current optimiser.param
            for _ in range(num_iter):
                with torch.enable_grad():
                    # evaluate L'_M at the current point of the SGLD chain (including gradients)
                    loss = closure()

                for group_index, group in enumerate(self.optimizer.param_groups):
                    for param_index, w_prime in enumerate(group["params"]):
                        # Center of the prior p(w)
                        w0 = param_groups[group_index]["params"][param_index]

                        # - \beta N \grad L'_N(w) with L'_M(w) as a surrogate for L'_N(w)
                        dx_prime = -w_prime.grad.data / np.log(self.total_train) * self.total_train

                        # \grad logp(w) - N \grad L'_N(w)
                        dx_prime.add_(w_prime.data - w0.data, alpha=-gamma)

                        # w' = epsilon_t / 2 ( \grad logp(w) - N \grad L'_N(w) )
                        w_prime.data.add_(dx_prime, alpha=epsilon / 2)
                        gaussian_noise = torch.empty_like(w_prime)
                        gaussian_noise.normal_()

                        # w' = epsilon_t / 2 ( \grad logp(w) - N \grad L'_N(w) ) + eta_t
                        w_prime.data.add_(gaussian_noise, alpha=(np.sqrt(epsilon)))

                Lms.append(loss)
            
            reset_chain()

        # E_w^\beta[ L'_M ] where the expectation over the posterior is approximated by SGLD
        Ew_Lm = sum(Lms) / len(Lms) 

        # N E_w^\beta[ L'_M ]
        local_free_energy = self.total_train * Ew_Lm 

        reset_chain()

        return local_free_energy