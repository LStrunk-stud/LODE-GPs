# gp/kernels.py
from typing import Callable, Dict, List, Sequence
import gpytorch

# ---------- registry ----------
_REGISTRY: Dict[str, Callable[..., gpytorch.kernels.Kernel]] = {}

def register(name: str) -> Callable[[Callable[..., gpytorch.kernels.Kernel]], Callable[..., gpytorch.kernels.Kernel]]:
    """Decorator that adds a builder to the kernel registry."""
    def _decorator(fn):
        _REGISTRY[name] = fn
        return fn
    return _decorator

def build(name: str, *, active_dims: Sequence[int] | None = None, weights=None) -> gpytorch.kernels.Kernel:
    """Factory: return a kernel instance by symbolic name."""
    try:
        return _REGISTRY[name](active_dims=active_dims, weights=weights)
    except KeyError as e:
        raise ValueError(f"Kernel '{name}' is not registered. "
                         f"Available: {list(_REGISTRY)}") from e

def available() -> List[str]:
    """List of all implemented kernels (for CLI / docs / tests)."""
    return list(_REGISTRY.keys())


# Comment on ordering: 
# Globarlly, we use the following order:
# 1D
# Constant times 1D
# Combinations of 1D kernels (size 2)
# Combinations of 1D kernels (size 3)
# ...
# Multi-input kernels

# Locally the ordering is alphabetically, + before *
# Inside a kernel name, the order is also alphabetically

##### 1D base

@register("LIN")
def _lin(*, active_dims=None, **_):
    return gpytorch.kernels.LinearKernel(active_dims=active_dims)

@register("MAT32")
def _mat32(*, active_dims=None, **_):
    return gpytorch.kernels.MaternKernel(nu=1.5, active_dims=active_dims)

@register("MAT52")
def _mat52(*, active_dims=None, **_):
    return gpytorch.kernels.MaternKernel(nu=2.5, active_dims=active_dims)

@register("PER")
def _per(*, active_dims=None, **_):
    return gpytorch.kernels.PeriodicKernel(active_dims=active_dims)

@register("RQ")
def _rq(*, active_dims=None, **_):
    return gpytorch.kernels.RQKernel(active_dims=active_dims)

@register("SE")                
def _se(*, active_dims=None, **_):
    return gpytorch.kernels.RBFKernel(active_dims=active_dims)

###### Constant times base

@register("C*LIN")
def _c_lin(*, active_dims=None, **_):
    return gpytorch.kernels.ScaleKernel(_lin(active_dims=active_dims))

@register("C*PER")
def _c_per(*, active_dims=None, **_):
    return gpytorch.kernels.ScaleKernel(_per(active_dims=active_dims))

@register("C*SE")
def _c_se(*, active_dims=None, **_):
    return gpytorch.kernels.ScaleKernel(_se(active_dims=active_dims))

@register("C*C*SE")
def _c_c_se(*, active_dims=None, **_):
    return gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.ScaleKernel(_se(active_dims=active_dims)))

####### Combinations of base kernels (size 2)

@register("LIN*PER")
def _lin_times_per(*, active_dims=None, **_):
    return _lin(active_dims=active_dims) * \
           _per(active_dims=active_dims)

@register("LIN*SE")
def _lin_times_se(*, active_dims=None, **_):
    return _lin(active_dims=active_dims) * \
           _se(active_dims=active_dims)

@register("MAT32+MAT52")
def _mat32_plus_mat52(*, active_dims=None, **_):
    return _mat32(active_dims=active_dims) + \
           _mat52(active_dims=active_dims)

@register("MAT32+PER")
def _mat32_plus_per(*, active_dims=None, **_):
    return _mat32(active_dims=active_dims) + \
           _per(active_dims=active_dims)

@register("MAT32*PER")
def _mat32_times_per(*, active_dims=None, **_):
    return _mat32(active_dims=active_dims) * \
           _per(active_dims=active_dims)

@register("MAT32+SE")
def _mat32_plus_se(*, active_dims=None, **_):
    return _mat32(active_dims=active_dims) + \
           _se(active_dims=active_dims)

@register("MAT32*SE")
def _mat32_times_se(*, active_dims=None, **_):
    return _mat32(active_dims=active_dims) * \
           _se(active_dims=active_dims)

@register("MAT52*PER")
def _mat52_times_per(*, active_dims=None, **_):
    return _mat52(active_dims=active_dims) * \
           _per(active_dims=active_dims)

@register("MAT52+SE")
def _mat52_plus_se(*, active_dims=None, **_):
    return _mat52(active_dims=active_dims) + \
           _se(active_dims=active_dims)

@register("PER*SE")
def _per_times_se(*, active_dims=None, **_):
    return _per(active_dims=active_dims) * \
           _se(active_dims=active_dims)

@register("PER+C*SE")
def _per_c_se(*, active_dims=None, **_):
    return _per(active_dims=active_dims) * \
           gpytorch.kernels.ScaleKernel(_se(active_dims=active_dims))        

@register("SE+SE")
def _se_plus_se(*, active_dims=None, **_):
    return _se(active_dims=active_dims) + \
           _se(active_dims=active_dims)

@register("SE*SE")
def _se_times_se(*, active_dims=None, **_):
    return _se(active_dims=active_dims) * \
           _se(active_dims=active_dims)

######## Combinations of base kernels (size 3)

@register("SE+SE+SE")
def _se_plus_se_plus_se(*, active_dims=None, **_):
    return _se(active_dims=active_dims) + \
           _se(active_dims=active_dims) + \
           _se(active_dims=active_dims)

@register("MAT32+(MAT52*PER)")
def _mat32_plus_mat52_times_per(*, active_dims=None, **_):
    return _mat32(active_dims=active_dims) + \
           (_mat52(active_dims=active_dims) * \
           _per(active_dims=active_dims))

@register("PER*(SE+RQ)")
def _se_plus_rq_times_per(*, active_dims=None, **_):
    return _per(active_dims=active_dims) * \
           (_se(active_dims=active_dims) +
            _rq(active_dims=active_dims))

###############################################################################
############################### MI Kernels ####################################
###############################################################################


@register("[LIN; LIN]")
def _lin_lin(*, active_dims=None, **_):
    # active_dims is a 2-tuple like ([0], [1]) you pass in the model
    k0 = _lin(active_dims=active_dims[0])
    k1 = _lin(active_dims=active_dims[1])
    return gpytorch.kernels.AdditiveStructureKernel(k0 + k1, num_dims=2)
        
@register("[LIN;* LIN]")
def _lin_lin_times(*, active_dims=None, **_):
    # active_dims is a 2-tuple like ([0], [1]) you pass in the model
    k0 = _lin(active_dims=active_dims[0])
    k1 = _lin(active_dims=active_dims[1])
    return k0 * k1

@register("[LIN; SE]")
def _lin_se(*, active_dims=None, **_):
    # active_dims is a 2-tuple like ([0], [1]) you pass in the model
    k0 = _lin(active_dims=active_dims[0])
    k1 = _se(active_dims=active_dims[1])
    return gpytorch.kernels.AdditiveStructureKernel(k0 + k1, num_dims=2)

@register("[SE;* 1]")
def _se_one_times(*, active_dims=None, **_):
    return gpytorch.kernels.RBFKernel(active_dims=active_dims[0])

@register("[PER;* 1]")
def _per_one_times(*, active_dims=None, **_):
    return gpytorch.kernels.PeriodicKernel(active_dims=active_dims[0])

@register("[SE; SE]")        # two independent SE kernels
def _se_se(*, active_dims=None, **_):
    # active_dims is a 2-tuple like ([0], [1]) you pass in the model
    k0 = _se(active_dims=active_dims[0])
    k1 = _se(active_dims=active_dims[1])
    return gpytorch.kernels.AdditiveStructureKernel(k0 + k1, num_dims=2)

@register("[SE;* SE]")        # two independent SE kernels, one of them is constant
def _se_se_times(*, active_dims=None, **_):
    # active_dims is a 2-tuple like ([0], [1]) you pass in the model
    k0 = _se(active_dims=active_dims[0])
    k1 = _se(active_dims=active_dims[1])
    return k0 * k1

@register("[SE+SE; 1]")        
def _se_plus_se_one(*, active_dims=None, **_):
    # active_dims is a 2-tuple like ([0], [1]) you pass in the model
    k0 = _se_plus_se(active_dims=active_dims[0])
    return gpytorch.kernels.AdditiveStructureKernel(k0, num_dims=2)

@register("[SE+SE; 1]_noadd")        
def _se_plus_se_one_noadd(*, active_dims=None, **_):
    # active_dims is a 2-tuple like ([0], [1]) you pass in the model
    k0 = _se_plus_se(active_dims=active_dims[0])
    return k0

@register("[SE; LIN]")
def _se_lin(*, active_dims=None, **_):
    # active_dims is a 2-tuple like ([0], [1]) you pass in the model
    k0 = _se(active_dims=active_dims[0])
    k1 = _lin(active_dims=active_dims[1])
    return gpytorch.kernels.AdditiveStructureKernel(k0 + k1, num_dims=2)
