import gpytorch
import torch
from helpers.example_kernels import build


class _BaseExactGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel_name="SE",
                 weights=None, active_dims=None):
        super(_BaseExactGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = build(kernel_name,
                                  active_dims=active_dims,
                                  weights=weights) 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

class _BaseDataGP(_BaseExactGP):

    def sample(self, x, n_samples=1, from_likelihood=False, prior_mode=False):
        with torch.no_grad(), gpytorch.settings.prior_mode(prior_mode):
            was_training = self.training

            self.eval()
            self.likelihood.eval()

            draw = (self.likelihood(self(x)) if from_likelihood else self(x)).sample_n(n_samples)
            if was_training:  # restore
                self.train()
                self.likelihood.train()
            return draw.detach()
        

class ExactGPModel(_BaseExactGP):
    pass

class DataGPModel(_BaseDataGP):
    pass