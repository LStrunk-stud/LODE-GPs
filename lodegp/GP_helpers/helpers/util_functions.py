import itertools
import torch
import numpy as np


# An empirically derived prior for the parameters of the kernels
informed_prior_dict = {'SE': {'raw_lengthscale' : {"mean": -0.21221139138922668 , "std":1.8895426067756804}},
                  'MAT52': {'raw_lengthscale' :{"mean": 0.7993038925994188, "std":2.145122566357853 } },
                  'MAT32': {'raw_lengthscale' :{"mean": 1.5711054238673443, "std":2.4453761235991216 } },
                  'RQ': {'raw_lengthscale' :{"mean": -0.049841950913676276, "std":1.9426354614713097 },
                          'raw_alpha' :{"mean": 1.882148553921053, "std":3.096431944989054 } },
                  'PER':{'raw_lengthscale':{"mean": 0.7778461197268618, "std":2.288946656544974 },
                          'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 } },
                  'LIN':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
                  'AFF':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
                  'c':{'raw_outputscale':{"mean": -1.6253091096349706, "std":2.2570021716661923 } },
                  'noise': {'raw_noise':{"mean": -3.51640656386717, "std":3.5831320474767407 }}}


# Credit for this code goes to https://github.com/JanHuewel
def get_string_representation_of_kernel(kernel_expression):
    if kernel_expression._get_name() == "AdditiveKernel":
        s = ""
        for k in kernel_expression.kernels:
            s += get_string_representation_of_kernel(k) + " + "
        return "(" + s[:-3] + ")"
    elif kernel_expression._get_name() == "AdditiveStructureKernel":
        return get_string_representation_of_kernel(kernel_expression.base_kernel)
    elif kernel_expression._get_name() == "ProductKernel":
        s = ""
        for k in kernel_expression.kernels:
            s += get_string_representation_of_kernel(k) + " * "
        return "(" + s[:-3] + ")"
    elif kernel_expression._get_name() == "ScaleKernel":
        return f"(c * {get_string_representation_of_kernel(kernel_expression.base_kernel)})"
    elif kernel_expression._get_name() == "RBFKernel":
        return "SE"
    elif kernel_expression._get_name() == "LinearKernel":
        return "LIN"
    elif kernel_expression._get_name() == "PeriodicKernel":
        return "PER"
    elif kernel_expression._get_name() == "MaternKernel":
        if kernel_expression.nu == 1.5:
            return "MAT32"
        elif kernel_expression.nu == 2.5:
            return "MAT52"
        else:
            raise "shit"
    elif kernel_expression._get_name() == "RQKernel":
        return "RQ"
    elif kernel_expression._get_name() == "AffineKernel":
        return "AFF"
    else:
        return kernel_expression._get_name()


def sample_value(shape, bounds, dist_type):
    """Sample values from a given distribution type."""
    if dist_type == "log-uniform":
        log_bounds = np.log(bounds)
        return torch.tensor(
            np.exp(np.random.uniform(*log_bounds, size=shape)), dtype=torch.float32
        )
    elif dist_type == "log":
        log_bounds = np.log(bounds)
        return torch.tensor(
            np.random.uniform(*log_bounds, size=shape), dtype=torch.float32
        )
    elif dist_type == "uniform":
        return torch.tensor(
            np.random.uniform(*bounds, size=shape), dtype=torch.float32
        )
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


def match_lode_parameter_spec(param_name, kernel_param_specs, default_bounds, default_type):
    """
    Match a LODE_Kernel parameter name to a kernel_param_specs entry.

    Looks for:
    - exact match
    - base match (e.g., "lengthscale" in "lengthscale_2")
    """
    if ("LODE_Kernel", param_name) in kernel_param_specs:
        spec = kernel_param_specs[("LODE_Kernel", param_name)]
        return spec.get("bounds", default_bounds), spec.get("type", default_type)

    for base_key in ["lengthscale", "signal_variance"]:
        if base_key in param_name:
            if ("LODE_Kernel", base_key) in kernel_param_specs:
                spec = kernel_param_specs[("LODE_Kernel", base_key)]
                return spec.get("bounds", default_bounds), spec.get("type", default_type)

    return default_bounds, default_type


def reparameterize_model(model, theta):
    for model_param, sampled_param in zip(model.parameters(), theta):
        model_param.data = torch.full_like(model_param.data, float(sampled_param))


def fixed_reinit(model, parameters: torch.tensor) -> None:
    for i, (param, value) in enumerate(zip(model.parameters(), parameters)):
        param.data = torch.full_like(param.data, value)




#def prior_distribution(model, uninformed=False):
#    uninformed_prior_dict = {'SE': {'raw_lengthscale' : {"mean": 0. , "std":10.}},
#                  'MAT52': {'raw_lengthscale' :{"mean": 0., "std":10. } },
#                  'MAT32': {'raw_lengthscale' :{"mean": 0., "std":10. } },
#                  'RQ': {'raw_lengthscale' :{"mean": 0., "std":10. },
#                          'raw_alpha' :{"mean": 0., "std":10. } },
#                  'PER':{'raw_lengthscale':{"mean": 0., "std":10. },
#                          'raw_period_length':{"mean": 0., "std":10. } },
#                  'LIN':{'raw_variance' :{"mean": 0., "std":10. } },
#                  'AFF':{'raw_variance' :{"mean": 0., "std":10. } },
#                  'c':{'raw_outputscale':{"mean": 0., "std":10. } },
#                  'noise': {'raw_noise':{"mean": 0., "std":10. }}}
#
#    # TODO de-spaghettize this once the priors are coded properly
#    prior_dict = {'SE': {'raw_lengthscale' : {"mean": -0.21221139138922668 , "std":1.8895426067756804}},
#                  'MAT52': {'raw_lengthscale' :{"mean": 0.7993038925994188, "std":2.145122566357853 } },
#                  'MAT32': {'raw_lengthscale' :{"mean": 1.5711054238673443, "std":2.4453761235991216 } },
#                  'RQ': {'raw_lengthscale' :{"mean": -0.049841950913676276, "std":1.9426354614713097 },
#                          'raw_alpha' :{"mean": 1.882148553921053, "std":3.096431944989054 } },
#                  'PER':{'raw_lengthscale':{"mean": 0.7778461197268618, "std":2.288946656544974 },
#                          'raw_period_length':{"mean": 0.6485334993738499, "std":0.9930632050553377 } },
#                  'LIN':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
#                  'AFF':{'raw_variance' :{"mean": -0.8017903983055685, "std":0.9966569921354465 } },
#                  'c':{'raw_outputscale':{"mean": -1.6253091096349706, "std":2.2570021716661923 } },
#                  'noise': {'raw_noise':{"mean": -3.51640656386717, "std":3.5831320474767407 }}}
#    #prior_dict = {"SE": {"raw_lengthscale": {"mean": 0.891, "std": 2.195}},
#    #              "MAT": {"raw_lengthscale": {"mean": 1.631, "std": 2.554}},
#    #              "PER": {"raw_lengthscale": {"mean": 0.338, "std": 2.636},
#    #                      "raw_period_length": {"mean": 0.284, "std": 0.902}},
#    #              "LIN": {"raw_variance": {"mean": -1.463, "std": 1.633}},
#    #              "c": {"raw_outputscale": {"mean": -2.163, "std": 2.448}},
#    #              "noise": {"raw_noise": {"mean": -1.792, "std": 3.266}}}
#
#    if uninformed:
#        prior_dict = uninformed_prior_dict
#
#
#    variances_list = list()
#    debug_param_name_list = list()
#    theta_mu = list()
#    params = None 
#    covar_string = gsr(model.covar_module)
#    covar_string = covar_string.replace("(", "")
#    covar_string = covar_string.replace(")", "")
#    covar_string = covar_string.replace(" ", "")
#    covar_string = covar_string.replace("PER", "PER+PER")
#    covar_string = covar_string.replace("RQ", "RQ+RQ")
#    covar_string_list = [s.split("*") for s in covar_string.split("+")]
#    covar_string_list.insert(0, ["LIKELIHOOD"])
#    covar_string_list = list(itertools.chain.from_iterable(covar_string_list))
#    both_PER_params = False
#    for (param_name, param), cov_str in zip(model.named_parameters(), covar_string_list):
#        if params == None:
#            params = param
#        else:
#            if len(param.shape)==0:
#                params = torch.cat((params,param.unsqueeze(0)))
#            elif len(param.shape)==1:
#                params = torch.cat((params,param))
#            else:
#                params = torch.cat((params,param.squeeze(0)))
#        debug_param_name_list.append(param_name)
#        curr_mu = None
#        curr_var = None
#        # First param is (always?) noise and is always with the likelihood
#        if "likelihood" in param_name:
#            curr_mu = prior_dict["noise"]["raw_noise"]["mean"]
#            curr_var = prior_dict["noise"]["raw_noise"]["std"]
#        else:
#            if (cov_str == "PER" or cov_str == "RQ") and not both_PER_params:
#                curr_mu = prior_dict[cov_str][param_name.split(".")[-1]]["mean"]
#                curr_var = prior_dict[cov_str][param_name.split(".")[-1]]["std"]
#                both_PER_params = True
#            elif (cov_str == "PER" or cov_str == "RQ") and both_PER_params:
#                curr_mu = prior_dict[cov_str][param_name.split(".")[-1]]["mean"]
#                curr_var = prior_dict[cov_str][param_name.split(".")[-1]]["std"]
#                both_PER_params = False
#            else:
#                try:
#                    curr_mu = prior_dict[cov_str][param_name.split(".")[-1]]["mean"]
#                    curr_var = prior_dict[cov_str][param_name.split(".")[-1]]["std"]
#                except Exception as E:
#                    import pdb
#                    #pdb.set_trace()
#                    prev_cov = cov_str
#        theta_mu.append(curr_mu)
#        variances_list.append(curr_var)
#    theta_mu = torch.tensor(theta_mu)
#    theta_mu = theta_mu.unsqueeze(0).t()
#    sigma = torch.diag(torch.Tensor(variances_list))
#    variance = sigma@sigma
#    return theta_mu, variance
 

#def log_normalized_prior(model, theta_mu=None, variance=None, uninformed=False):
#    theta_mu, variance = prior_distribution(model, uninformed=uninformed) if theta_mu is None or variance is None else (theta_mu, variance)
#    prior = torch.distributions.MultivariateNormal(theta_mu.t(), variance)
#
#    params = None
#    for (param_name, param) in model.named_parameters():
#        if params == None:
#            params = param
#        else:
#            if len(param.shape)==0:
#                params = torch.cat((params,param.unsqueeze(0)))
#            elif len(param.shape)==1:
#                params = torch.cat((params,param))
#            else:
#                params = torch.cat((params,param.squeeze(0)))
# 
#    # for convention reasons I'm diving by the number of datapoints
#    log_prob = prior.log_prob(params) / len(*model.train_inputs)
#    return log_prob.squeeze(0)





def randomize_model_hyperparameters(
    model,
    param_specs=None,
    kernel_param_specs=None,
    default_bounds=(0.1, 1.0),
    default_type="log",
    verbose=False
):
    param_specs = param_specs or {}
    kernel_param_specs = kernel_param_specs or {}

    # Step 1: Get kernel types in order
    kernel_types = get_full_kernels_in_kernel_expression(model.covar_module)
    kernel_index = 0

    for name, param in model.named_hyperparameters():
        shape = param.shape

        # Case 1: Full-name match
        if name in param_specs:
            spec = param_specs[name]
            bounds = spec.get("bounds", default_bounds)
            dist_type = spec.get("type", default_type)
            kernel_type = name

        # Case 2: LODE_Kernel param in ParameterDict
        elif "LODE_Kernel" in kernel_types:
            # Extract the innermost param name
            local_param_name = name.split(".")[-1]
            bounds, dist_type = match_lode_parameter_spec(
                local_param_name, kernel_param_specs, default_bounds, default_type
            )
            kernel_type = "LODE_Kernel"

        # Case 3: Kernel-based fallback (sequential assignment)
        else:
            local_param_name = name.split(".")[-1]
            if kernel_index < len(kernel_types):
                kernel_type = kernel_types[kernel_index]
                kernel_index += 1
                spec = kernel_param_specs.get((kernel_type, local_param_name), {})
                bounds = spec.get("bounds", default_bounds)
                dist_type = spec.get("type", default_type)
            else:
                kernel_type = "<unknown>"
                bounds = default_bounds
                dist_type = default_type

        # Sample and assign
        new_value = sample_value(shape, bounds, dist_type)
        with torch.no_grad():
            param.copy_(new_value)

        if verbose:
            print(f"[Reinit] {name} ← {new_value.cpu().numpy()} (kernel: {kernel_type}, dist: {dist_type}, bounds: {bounds})")


# Credit for this code goes to https://github.com/JanHuewel
def get_full_kernels_in_kernel_expression(kernel_expression):
    """
    returns list of all base kernels in a kernel expression
    """
    kernel_list = list()
    if kernel_expression == None:
        return kernel_list
    if hasattr(kernel_expression, "kernels"):
        for kernel in kernel_expression.kernels:
            kernel_list.extend(get_full_kernels_in_kernel_expression(kernel))
    elif kernel_expression._get_name() in ["ScaleKernel", "GridKernel"]:
        kernel_list.extend([kernel_expression._get_name()])
        kernel_list.extend(get_full_kernels_in_kernel_expression(
            kernel_expression.base_kernel))
    elif kernel_expression._get_name() in ["AdditiveStructureKernel"]:
        kernel_list.extend(get_full_kernels_in_kernel_expression(
            kernel_expression.base_kernel))
    else:
        kernel_list.append(kernel_expression._get_name())
    return kernel_list



def get_param_spec(
    param_name,
    kernel_name,
    param_specs=None,
    kernel_param_specs=None,
    default_spec=None
):
    """
    Resolves a parameter spec using full name match, kernel+param, or fuzzy match for LODE_Kernel.

    Returns:
    - spec dict (e.g., {"bounds": (a, b), "type": ..., "mean": ..., "variance": ...})
    """
    param_specs = param_specs or {}
    kernel_param_specs = kernel_param_specs or {}
    default_spec = default_spec or {}

    # 1. Full name override
    if param_name in param_specs:
        return param_specs[param_name]

    # 2. Kernel + parameter
    if (kernel_name, param_name) in kernel_param_specs:
        return kernel_param_specs[(kernel_name, param_name)]

    # 3. Fuzzy base-key match for LODE_Kernel
    if kernel_name == "LODE_Kernel":
        for base_key in ["lengthscale", "signal_variance"]:
            if base_key in param_name:
                if (kernel_name, base_key) in kernel_param_specs:
                    return kernel_param_specs[(kernel_name, base_key)]

    # 4. Fallback
    return default_spec


def prior_distribution(
    model,
    param_specs=None,
    kernel_param_specs=None,
    default_mean=0.0,
    default_std=10.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collects prior parameters (mean, variance) for all named hyperparameters in the model.

    Returns:
    - List of tuples: (param_name, tensor, mean, variance)
    """
    mean_values = []
    var_values = []
    kernel_types = get_full_kernels_in_kernel_expression(model.covar_module)
    kernel_index = 0

    for name, _ in model.named_hyperparameters():
        if not param_specs is None:
            if name in param_specs:
                kernel_type = "<explicit>"
            elif "LODE_Kernel" in kernel_types:
                kernel_type = "LODE_Kernel"
            else:
                kernel_type = kernel_types[kernel_index] if kernel_index < len(kernel_types) else "<unknown>"
                kernel_index += 1
        else:
            kernel_type = kernel_types[kernel_index] if kernel_index < len(kernel_types) else "<unknown>"
            kernel_index += 1

        spec = get_param_spec(
            name,
            kernel_type,
            param_specs=param_specs,
            kernel_param_specs=kernel_param_specs,
            default_spec={"mean": default_mean, "std": default_std}
        )
        
        # Catching special cases where parameters are vectors
        # MultitaskGPs have an independent noise for each task
        if name == "likelihood.raw_task_noises":
            mean_values.extend([spec["mean"] for _ in range(model.num_tasks)])
            var_values.extend([spec["std"]**2 for _ in range(model.num_tasks)])
        # These are also ARD Kernels, which I usually don't care about, that have the same issue
        else:
            mean_values.append(spec["mean"])
            var_values.append(spec["std"]**2)

    return torch.distributions.MultivariateNormal(torch.tensor(mean_values), torch.diag(torch.tensor(var_values)))


def extract_model_parameters(model):
    params = None
    for (_, param) in model.named_parameters():
        if params == None:
            params = param
        else:
            if len(param.shape)==0:
                params = torch.cat((params, param.unsqueeze(0)))
            elif len(param.shape)==1:
                params = torch.cat((params, param))
            else:
                params = torch.cat((params, param.squeeze(0)))
    return params

def log_normalized_prior(model, param_specs, kernel_param_specs, theta_mu=None, variance=None, prior=None):
    if prior is None:
        theta_mu, variance = prior_distribution(model, param_specs=param_specs, kernel_param_specs=kernel_param_specs) if theta_mu is None or variance is None else (theta_mu, variance)
        if not type(theta_mu) == torch.Tensor:
            theta_mu = torch.tensor(theta_mu)
        if not type(variance) == torch.Tensor: 
            variance = torch.tensor(variance)
            variance = torch.diag(variance)
        elif not (variance.ndim == 2 and variance.size(0) == variance.size(1)):
            variance = torch.diag(variance)

        prior = torch.distributions.MultivariateNormal(theta_mu.t(), variance)

    params = extract_model_parameters(model)

    # for convention reasons I'm dividing by the number of datapoints
    log_prob = prior.log_prob(params) / len(*model.train_inputs)
    return log_prob.squeeze(0)

# Find all points inside the confidence ellipse
def percentage_inside_ellipse(mu, K, points, sigma_level=2):
    L = np.linalg.cholesky(K)
    threshold = sigma_level ** 2
    count = 0
    for point in points:
        res = np.array(point - mu) @ np.linalg.inv(L)
        if res @ res <= threshold:
            count += 1
    return count / len(points)

def central_difference(f, x, h=1e-2, order=1, precision = 6):
    if order == 0:
        return f(x)
    elif order == 1:
        if precision == 2:
            return (f(x + h) - f(x - h)) / (h)
        elif precision == 6:
            return (float(1/60)*f(x + 3*h) - float(3/20)*f(x + 2*h) + 0.75*f(x + h) - 0.75*f(x - h) + float(3/20)*f(x - 2*h) - float(1/60)*f(x - 3*h))/(h)
    elif order == 2:
        if precision == 2:
            return (f(x + h) - 2 * f(x) + f(x - h)) / (h**2)
        elif precision == 6:
            return (1/90*f(x+3*h) - 3/20*f(x+2*h) + 1.5*f(x+h) - 49/18*f(x) + 1.5*f(x-h) - 3/20*f(x-2*h) + 1/90*f(x- 3*h))/(h**2)
            #return (-float(1/56)*f(x + 4*h) + float(8/315)*f(x+ 3*h) - 1/5* f(x + 2*h) + 8/5*f(x+h) - 205/72*f(x) + 8/5*f(x-h) - 1/5*f(x-2*h) + 8/315*f(x-3*h) - 1/560*f(x-4*h))/(h**2)
    elif order == 3:
        return (0.5*f(x + 2*h) - f(x + h) + f(x - h) - 0.5*f(x - 2*h))/(h**3)

def extract_differential_polynomial_terms(expr, diff_var, var_dict):
    # Only if I calculate the ODE satisfaction errors require the sagemath library
    from sage.all import sage_eval
    import sage
    #https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
    from sage.calculus.var import var
    """
    Parses a polynomial expression in the differential operator variable (e.g., x),
    and returns a dict mapping derivative order to coefficient.

    Parameters:
    - expr: Sage symbolic expression (e.g., x^2 + 981/100)
    - diff_var: the Sage variable representing the differential operator (e.g., x)

    Returns:
    - dict: {order: coefficient}  (e.g., {0: 981/100, 2: 1})
    """
    expr = sage_eval(str(expr), locals=var_dict)
    if type(expr) is not sage.symbolic.expression.Expression:
        return {0: expr}
    terms = expr.coefficients(diff_var)
    result = {}

    for term in terms:
        # each term is a tuple [coefficient, degree]
        coeff = term[0]
        degree = term[1]
        result[degree] = coeff

    return result


# Verify that the given data satisfies the given differential equation
def calculate_differential_equation_error_numeric(differential_eq, sage_locals, data_generating_functions, data, **kwargs):
    from sage.all import sage_eval
    import sage
    #https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
    from sage.calculus.var import var
    dx = kwargs.get("diff_var", var("x"))
    locals_values = kwargs.get("locals_values", {sage_locals[var_name] : 1.0 for var_name in sage_locals})
    # We know we that the channel count is equal to the number of tasks
    channel_values = [[] for _ in range(len(differential_eq))]
    differential_equation_error = None 
    functions = data_generating_functions
    for i, column in enumerate(differential_eq):
        # Each channel contains the polynom of differentials that is used on the respective channel
        # Dictionary of the form {order: coeff}
        coeff_dict = extract_differential_polynomial_terms(column, dx, sage_locals)
        for order, coeff in coeff_dict.items():
            try:
                coeff = coeff.subs(locals_values)
                diff_approx = central_difference(functions, data, order=order)[:, i]
                if differential_equation_error is None:
                    differential_equation_error = float(coeff)*diff_approx
                else:
                    differential_equation_error += float(coeff)*diff_approx
            except Exception as e:
                print(coeff)
                print(e)
    return differential_equation_error

# Verify that the functions satisfy the given differential equation
def calculate_differential_equation_error_symbolic(functions, differential_eq, sage_locals, **kwargs):
    from sage.all import sage_eval
    import sage
    #https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
    from sage.calculus.var import var
    # We know we that the channel count is equal to the number of tasks
    dx = kwargs.get("diff_var", var("x"))
    differential_equation_error = 0
    for i, column in enumerate(differential_eq):
        # Each channel contains the polynom of differentials that is used on the respective channel
        # Dictionary of the form {order: coeff}
        coeff_dict = extract_differential_polynomial_terms(column, dx, sage_locals)
        for order, coeff in coeff_dict.items():
            differential_equation_error += coeff*functions[i].diff(dx, int(order))
    return differential_equation_error
