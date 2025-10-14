import copy
from .util_functions import log_normalized_prior, get_full_kernels_in_kernel_expression, randomize_model_hyperparameters
import gpytorch
import numpy as np
from pygranso.pygranso import pygranso
from pygranso.pygransoStruct import pygransoStruct
from pygranso.private.getNvar import getNvarTorch
import torch


kernel_parameter_priors = {
    ("RBFKernel", "lengthscale"): {"mean": 0.0, "std": 10.0}, 
    ("MaternKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("LinearKernel", "variance"): {"mean": 0.0, "std": 10.0},
    ("AffineKernel", "variance"): {"mean": 0.0, "std": 10.0},
    ("RQKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("RQKernel", "alpha"): {"mean": 0.0, "std": 10.0},
    ("CosineKernel", "period_length"): {"mean": 0.0, "std": 10.0},
    ("PeriodicKernel", "lengthscale"): {"mean": 0.0, "std": 10.0},
    ("PeriodicKernel", "period_length"): {"mean": 0.0, "std": 10.0},
    ("ScaleKernel", "outputscale"): {"mean": 0.0, "std": 10.0},
    ("LODE_Kernel", "signal_variance_2_0"): {"mean": 0.0, "std": 10.0},  # full match
    ("LODE_Kernel", "lengthscale"): {"mean": 0.0, "std": 10.0},           # base fallback
}


parameter_priors = {
    "likelihood.raw_task_noises": {"mean": 0.0, "std": 10.0},
    "likelihood.raw_noise": {"mean": 0.0, "std": 10.0}
}


kernel_param_specs = {
    ("RBFKernel", "lengthscale"): {"bounds": (1e-1, 5.0)}, # add ', "type": "uniform"},' # to use uniform distribution
    ("MaternKernel", "lengthscale"): {"bounds": (1e-1, 1.0)},
    ("LinearKernel", "variance"): {"bounds": (1e-1, 1.0)},
    ("AffineKernel", "variance"): {"bounds": (1e-1, 1.0)},
    ("RQKernel", "lengthscale"): {"bounds": (1e-1, 1.0)},
    ("RQKernel", "alpha"): {"bounds": (1e-1, 1.0)},
    ("CosineKernel", "period_length"): {"bounds": (1e-1, 10.0), "type": "uniform"},
    ("PeriodicKernel", "lengthscale"): {"bounds": (1e-1, 5.0)},
    ("PeriodicKernel", "period_length"): {"bounds": (1e-1, 10.0), "type": "uniform"},
    ("ScaleKernel", "outputscale"): {"bounds": (1e-1, 10.0)},
    #("LODE_Kernel", "signal_variance_2_0"): {"bounds": (0.05, 0.5)},  # full match
    ("LODE_Kernel", "signal_variance"): {"bounds": (1e-1, 10)},  # base
    ("LODE_Kernel", "lengthscale"): {"bounds": (1e-1, 5.0)},           
}


param_specs = {
    "likelihood.raw_task_noises": {"bounds": (1e-1, 1e-0)},
    "likelihood.raw_noise": {"bounds": (1e-1, 1e-0)}
}


def fixed_reinit(model, parameters: torch.tensor) -> None:
    for i, (param, value) in enumerate(zip(model.parameters(), parameters)):
        param.data = torch.full_like(param.data, value)



def adam_optimization(model, likelihood, train_x, train_y, **kwargs):

    random_restarts = kwargs.get("random_restarts", 5)
    MAP = kwargs.get("MAP", True)
    verbose = kwargs.get("verbose", False)
    iterations = kwargs.get("iterations", 100)
    lr = kwargs.get("lr", 0.1)

    # Define the negative log likelihood
    mll_fkt = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    uninformed = kwargs.get("uninformed", False)
    logarithmic_reinit = kwargs.get("logarithmic_reinit", False)
    model_parameter_prior = kwargs.get("model_parameter_prior", None)


    all_state_dicts_likelihoods_losses = []

    for restart in range(random_restarts):
        if verbose:
            print("---")
            print("start parameters: ", torch.nn.utils.parameters_to_vector(model.parameters()).detach())
        for i in range(iterations):
            optimizer.zero_grad()
            loss = -mll_fkt(model(train_x), train_y)

            if MAP:
                # log_normalized_prior is in metrics.py 
                log_p = log_normalized_prior(model, param_specs=parameter_priors, kernel_param_specs=kernel_parameter_priors, prior=model_parameter_prior)
                # negative scaled MAP
                loss -= log_p
            loss.backward()
            optimizer.step()

        if verbose:
            print(f"Restart {restart} : trained parameters: {list(model.named_parameters())}")

        all_state_dicts_likelihoods_losses.append((copy.deepcopy(model.state_dict()), copy.deepcopy(likelihood.state_dict()), loss))
        randomize_model_hyperparameters(model, param_specs=param_specs, kernel_param_specs=kernel_param_specs, verbose=verbose)

    for state_dict, likelihood_state_dict, loss in sorted(all_state_dicts_likelihoods_losses, key=lambda x: x[2]):
        model.load_state_dict(state_dict)
        likelihood.load_state_dict(likelihood_state_dict)
        try:
            loss = -mll_fkt(model(train_x), train_y)
            if MAP:
                log_p = log_normalized_prior(model, param_specs=parameter_priors, kernel_param_specs=kernel_parameter_priors, prior=model_parameter_prior)
                loss -= log_p
            if verbose:
                print(f"----")
                print(f"Final best parameters: {list(model.named_parameters())} w. loss: {loss} (smaller=better)")
                print(f"----")
            break
        except Exception:
            continue

    return loss, model, likelihood, None


# Define the PyGRANSO training loop
def granso_optimization(model, likelihood, train_x, train_y, **kwargs):
    """
    find optimal hyperparameters either by BO or by starting from random initial values multiple times, using an optimizer every time
    and then returning the best result
    """

    # I think this is very ugly to define the class inside the training function and then use a parameter from the function within the class scope. But we all need to make sacrifices...
    # Original class taken from https://ncvx.org/examples/A1_rosenbrock.html
    class HaltLog:
        def __init__(self):
            pass

        def haltLog(self, iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized,
                    ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level):

            # DON'T CHANGE THIS
            # increment the index/count
            self.index += 1

            # EXAMPLE:
            # store history of x iterates in a preallocated cell array
            self.x_iterates[restart].append(x)
            self.neg_loss[restart].append(penaltyfn_parts.f)
            self.tv[restart].append(penaltyfn_parts.tv)
            self.hessians[restart].append(get_BFGS_state_fn())

            # keep this false unless you want to implement a custom termination
            # condition
            halt = False
            return halt

        # Once PyGRANSO has run, you may call this function to get retreive all
        # the logging data stored in the shared variables, which is populated
        # by haltLog being called on every iteration of PyGRANSO.
        def getLog(self):
            # EXAMPLE
            # return x_iterates, trimmed to correct size
            log = pygransoStruct()
            log.x        = self.x_iterates
            log.neg_loss = self.neg_loss
            log.tv       = self.tv
            log.hessians = self.hessians
            #log = pygransoStruct()
            #log.x   = self.x_iterates[0:self.index]
            #log.f   = self.f[0:self.index]
            #log.tv  = self.tv[0:self.index]
            #log.hessians  = self.hessians[0:self.index]
            return log

        def makeHaltLogFunctions(self, restarts=1):
            # don't change these lambda functions
            halt_log_fn = lambda iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level: self.haltLog(iteration, x, penaltyfn_parts, d,get_BFGS_state_fn, H_regularized, ls_evals, alpha, n_gradients, stat_vec, stat_val, fallback_level)

            get_log_fn = lambda : self.getLog()

            # Make your shared variables here to store PyGRANSO history data
            # EXAMPLE - store history of iterates x_0,x_1,...,x_k

            # restart the index and empty the log
            self.index       = 0
            self.x_iterates  = [list() for _ in range(restarts)]
            self.neg_loss    = [list() for _ in range(restarts)]
            self.tv          = [list() for _ in range(restarts)]
            self.hessians    = [list() for _ in range(restarts)]

            # Only modify the body of logIterate(), not its name or arguments.
            # Store whatever data you wish from the current PyGRANSO iteration info,
            # given by the input arguments, into shared variables of
            # makeHaltLogFunctions, so that this data can be retrieved after PyGRANSO
            # has been terminated.
            #
            # DESCRIPTION OF INPUT ARGUMENTS
            #   iter                current iteration number
            #   x                   current iterate x
            #   penaltyfn_parts     struct containing the following
            #       OBJECTIVE AND CONSTRAINTS VALUES
            #       .f              objective value at x
            #       .f_grad         objective gradient at x
            #       .ci             inequality constraint at x
            #       .ci_grad        inequality gradient at x
            #       .ce             equality constraint at x
            #       .ce_grad        equality gradient at x
            #       TOTAL VIOLATION VALUES (inf norm, for determining feasibiliy)
            #       .tvi            total violation of inequality constraints at x
            #       .tve            total violation of equality constraints at x
            #       .tv             total violation of all constraints at x
            #       TOTAL VIOLATION VALUES (one norm, for L1 penalty function)
            #       .tvi_l1         total violation of inequality constraints at x
            #       .tvi_l1_grad    its gradient
            #       .tve_l1         total violation of equality constraints at x
            #       .tve_l1_grad    its gradient
            #       .tv_l1          total violation of all constraints at x
            #       .tv_l1_grad     its gradient
            #       PENALTY FUNCTION VALUES
            #       .p              penalty function value at x
            #       .p_grad         penalty function gradient at x
            #       .mu             current value of the penalty parameter
            #       .feasible_to_tol logical indicating whether x is feasible
            #   d                   search direction
            #   get_BFGS_state_fn   function handle to get the (L)BFGS state data
            #                       FULL MEMORY:
            #                       - returns BFGS inverse Hessian approximation
            #                       LIMITED MEMORY:
            #                       - returns a struct with current L-BFGS state:
            #                           .S          matrix of the BFGS s vectors
            #                           .Y          matrix of the BFGS y vectors
            #                           .rho        row vector of the 1/sty values
            #                           .gamma      H0 scaling factor
            #   H_regularized       regularized version of H
            #                       [] if no regularization was applied to H
            #   fn_evals            number of function evaluations incurred during
            #                       this iteration
            #   alpha               size of accepted size
            #   n_gradients         number of previous gradients used for computing
            #                       the termination QP
            #   stat_vec            stationarity measure vector
            #   stat_val            approximate value of stationarity:
            #                           norm(stat_vec)
            #                       gradients (result of termination QP)
            #   fallback_level      number of strategy needed for a successful step
            #                       to be taken.  See bfgssqpOptionsAdvanced.
            #
            # OUTPUT ARGUMENT
            #   halt                set this to true if you wish optimization to
            #                       be halted at the current iterate.  This can be
            #                       used to create a custom termination condition,
            return [halt_log_fn, get_log_fn]

    random_restarts = kwargs.get("random_restarts", 5)
    maxit = kwargs.get("maxit", 1000)
    model_parameter_prior = kwargs.get("model_parameter_prior", None)


    """
    # The call that comes from GRANSO
    user_halt = halt_log_fn(0, x, self.penaltyfn_at_x, np.zeros((n,1)),
                                        get_bfgs_state_fn, H_QP,
                                        1, 0, 1, stat_vec, self.stat_val, 0)
    """

    MAP = kwargs.get("MAP", True)
    double_precision = kwargs.get("double_precision", False)
    verbose = kwargs.get("verbose", False)

    # Define the negative log likelihood
    mll_fkt = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Set up the PyGRANSO optimizer
    opts = pygransoStruct()
    opts.torch_device = torch.device('cpu')
    nvar = getNvarTorch(model.parameters())
    opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)
    opts.opt_tol = float(1e-10)
    #opts.limited_mem_size = int(100)
    opts.limited_mem_size = 0
    opts.globalAD = True
    opts.double_precision = double_precision
    opts.quadprog_info_msg = False
    opts.print_level = int(0)
    opts.halt_on_linesearch_bracket = False
    opts.maxit = maxit
    mHLF_obj = HaltLog()
    [halt_log_fn, get_log_fn] = mHLF_obj.makeHaltLogFunctions(restarts=random_restarts)

    #  Set PyGRANSO's logging function in opts
    opts.halt_log_fn = halt_log_fn

    # Define the objective function
    def objective_function(model):
        output = model(train_x)
        try:
            # TODO PyGRANSO dying is a severe problem. as it literally exits the program instead of raising an error
            # negative scaled MLL
            loss = -mll_fkt(output, train_y)
        except Exception as E:
            print("LOG ERROR: Severe PyGRANSO issue. Loss is inf+0")
            print(f"LOG ERROR: {E}")
            loss = torch.tensor(np.finfo(np.float32).max, requires_grad=True) + torch.tensor(-10.0)
        if MAP:
            # log_normalized_prior is in metrics.py 
            log_p = log_normalized_prior(model, param_specs=parameter_priors, kernel_param_specs=kernel_parameter_priors, prior=model_parameter_prior)
            # negative scaled MAP
            loss -= log_p
        #print(f"LOG: {loss}")
        return [loss, None, None]

    all_state_dicts_likelihoods_losses = []

    for restart in range(random_restarts):
        if verbose:
            print("---")
            print("start parameters: ", opts.x0)
        # Train the model using PyGRANSO
        try:
            soln = pygranso(var_spec=model, combined_fn=objective_function, user_opts=opts)
            if verbose:
                print(f"Restart {restart} : trained parameters: {list(model.named_parameters())}")
        except Exception as e:
            print(e)
            import pdb
            pdb.set_trace()
            pass

        all_state_dicts_likelihoods_losses.append((copy.deepcopy(model.state_dict()), copy.deepcopy(likelihood.state_dict()), soln.final.f))
        randomize_model_hyperparameters(model, param_specs=param_specs, kernel_param_specs=kernel_param_specs, verbose=verbose)
        opts.x0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach().reshape(nvar,1)

    for state_dict, likelihood_state_dict, loss in sorted(all_state_dicts_likelihoods_losses, key=lambda x: x[2]):
        model.load_state_dict(state_dict)
        likelihood.load_state_dict(likelihood_state_dict)
        try:
            loss = -mll_fkt(model(train_x), train_y)
            if MAP:
                log_p = log_normalized_prior(model, param_specs=parameter_priors, kernel_param_specs=kernel_parameter_priors, prior=model_parameter_prior)
                loss -= log_p
            if verbose:
                print(f"----")
                print(f"Final best parameters: {list(model.named_parameters())} w. loss: {loss} (smaller=better)")
                print(f"----")
            break
        except Exception:
            continue
    
    # Return the trained model
    return loss, model, likelihood, get_log_fn()