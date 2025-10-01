#=======================================================================
# Imports
#=======================================================================
import gpytorch 
from sage.all import *
import sage
#https://ask.sagemath.org/question/41204/getting-my-own-module-to-work-in-sage/
from sage.calculus.var import var
from lodegp.kernels import *
import pprint
import torch
from lodegp.LODEGP import LODEGP
import numpy as np
from GP_helpers.helpers.util_functions import calculate_differential_equation_error_symbolic, calculate_differential_equation_error_numeric
import pytest

torch.set_default_dtype(torch.float64)

def symmetry_check(model):
    dx1, dx2 = var('dx1, dx2')
    symmetry_check_matrix = model.matrix_multiplication - matrix([[cell.substitute(dx1=dx2, dx2=dx1) for cell in row] for row in model.matrix_multiplication.T])
    # Symmetric up to numerical precision
    return all([all([cell < 1e-10 for cell in row]) for row in symmetry_check_matrix])

def eigval_check(model, train_x):
    covar_matrix = model(train_x).covariance_matrix
    eigvals = torch.linalg.eig(covar_matrix)[0]
    print("Eigenwerte (real):", eigvals)
    return all([eig.real > -1e-10 for eig in eigvals]), all([eig.imag < 1e-10 for eig in eigvals])

def test_eigenvalues():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Heating")
    eig_positive, eig_real = eigval_check(model, train_x)
    assert eig_positive 
    assert eig_real 
    
def test_symmetry_check():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Heating")
   # This matrix is suppossed to contain _only_ zeros
    assert symmetry_check(model)
    
   
def test_function():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
    A = matrix(SR, 1, 1, [1])
    x = var('x')
    model = LODEGP(train_x, train_y, likelihood, 1, A=A, sage_locals={'x': x})
    assert model.A == A

@pytest.mark.parametrize("base_kernel", ["SE_kernel", "Matern_kernel_32", "Matern_kernel_52"])
def test_heating(base_kernel):
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3)
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Heating", base_kernel=base_kernel)
    eig_positive, eig_real = eigval_check(model, train_x)
    assert eig_positive
    assert eig_real

@pytest.mark.parametrize("base_kernel", ["SE_kernel", "Matern_kernel_32", "Matern_kernel_52"])
def test_bipendulumkernel(base_kernel):
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3)
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Bipendulum", base_kernel=base_kernel)
    eig_positive, eig_real = eigval_check(model, train_x)
    assert eig_positive
    assert eig_real
    model.sage_locals = {'x': var('x')}
    
@pytest.mark.parametrize("base_kernel", ["SE_kernel", "Matern_kernel_32", "Matern_kernel_52"])
def test_threetank(base_kernel):
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(5, noise_constraint=gpytorch.constraints.Positive())
    model = LODEGP(train_x, train_y, likelihood, 5, ODE_type="Three tank", base_kernel=base_kernel)
    eig_positive, eig_real = eigval_check(model, train_x)
    assert eig_positive
    assert eig_real
    model.sage_locals = {'x': var('x')}

def test_calculate_differential_equation_error_symbolic():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3)
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Bipendulum")
    
    x = var('x')
    u1 = function('u1')(x)
    u2 = function('u2')(x)
    u3 = function('u3')(x)
    
    functions = [u1, u2, u3]  
    differential_eq = list(model.A.rows())  
    sage_locals = model.sage_locals


    result = calculate_differential_equation_error_symbolic(functions, differential_eq, sage_locals)
    print("Symbolischer ODE-Fehler:", result)






def test_calculate_differential_equation_error_numeric():
    train_x = torch.linspace(0, 1, 10)
    train_y = torch.linspace(0, 1, 10)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(3, noise_constraint=gpytorch.constraints.Positive())
    model = LODEGP(train_x, train_y, likelihood, 3, ODE_name="Bipendulum")
    differential_eq = list(model.A.rows())  
    sage_locals = model.sage_locals
    data = np.linspace(0, 2*np.pi, 100)
    data_generating_functions = [lambda x: np.sin(x), lambda x: np.cos(x)]
    result = calculate_differential_equation_error_numeric(differential_eq, sage_locals, data_generating_functions, data)
    print("Numerischer ODE-Fehler:", result)
  


def test_regex():
    test_expressions = [
    "e^(x+y)",                
    "e^z",                    
    "sin(x)^2 + cos(y)",      
    "sqrt(x + y)",            
    "exp(x)"   ,  
    "x^2"           
]
    for expr in test_expressions:
        transformed = replace_basic_operations(expr)
        print(f"Ersetzt : {transformed}")



