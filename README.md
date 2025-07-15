# adjointx

A JAX module for computing gradients through functionals constrained by equations using the adjoint method.

## Overview

`adjointx` provides efficient gradient computation for optimization problems of the form:

$$\min_m J(m) = D(u(m)) + R(m)$$

subject to the constraint:

$$F(u(m), m) = 0$$

where:
- $m$ is the control/parameter vector
- $u(m)$ is the state variable that depends on $m$ through the constraint
- $D(u)$ is the data loss term
- $R(m)$ is regularization
- $F(u, m) = 0$ represents the governing equations (PDEs, ODEs, Algebraic, etc.)

This formulation commonly arises in:
- **Inverse problems**: Inferring model parameters from observations
- **Optimal control**: Finding controls that minimize a cost subject to dynamics
- **PDE-constrained optimization**: Optimization problems governed by differential equations

## Mathematical Background

### The Adjoint Method

Direct differentiation of $J(m)$ with respect to $m$ requires computing $\frac{\partial u}{\partial m}$, which can be computationally expensive for high-dimensional parameter spaces. The adjoint method provides an efficient alternative.

Using the method of Lagrange multipliers, we form the Lagrangian:

$$\mathcal{L}(u, m, p) = D(u) + R(m) + p^T F(u, m)$$

The adjoint method exploits the fact that at the optimum, the constraint is satisfied ($F(u, m) = 0$), so:

$$\frac{dJ}{dm} = \frac{\partial \mathcal{L}}{\partial m}$$

The adjoint variable $p$ must be selected to satisfy the **adjoint equation**:

$$\frac{\partial \mathcal{L}}{\partial u} = 0 \implies \left(\frac{\partial F}{\partial u}\right)^T p = -\frac{\partial D}{\partial u}$$

Once $p$ is computed, the gradient is given by:

$$\frac{dJ}{dm} = \frac{\partial R}{\partial m} + p^T \frac{\partial F}{\partial m}$$

### Computational Advantages

- **Memory efficiency**: Requires solving only one additional linear system (the adjoint equation)
- **Computational cost**: $O(1)$ in the number of parameters, making it ideal for high-dimensional problems
- **Automatic differentiation**: JAX handles the computation of required Jacobians automatically

## Installation

```bash
uv add adjointx
```

Or for development:

```bash
git clone https://github.com/username/adjointx.git
cd adjointx
uv pip install -e .
```

## Quick Start

```python
import jax
import jax.numpy as jnp
from adjointx import construct_objective

# Define your forward problem F(u, m) = 0
def forward_operator(u, m):
    """Example: simple linear system Au = m"""
    A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    return A @ u - m

# Define your data loss D(u)
def data_loss(u):
    """Example: least squares data fitting"""
    u_obs = jnp.array([1.0, 2.0])  # observed data
    return 0.5 * jnp.sum((u - u_obs)**2)

# Define regularization R(m)
def regularization(m):
    """Example: L2 regularization"""
    return 0.01 * jnp.sum(m**2)

# Simple forward solver (you would use a more sophisticated one)
def simple_solver(forward_op, m):
    """Solve F(u, m) = 0 for u given m"""
    A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    return jnp.linalg.solve(A,m)

# Construct objective with adjoint-based gradients
objective = construct_objective(
    forward_operator,
    data_loss,
    regularization,
    simple_solver
)

# Use the objective function and its gradients
m = jnp.array([3.0, 4.0])  # initial parameters
J_m = objective(m)  # compute J(m)
grad_m = jax.grad(objective)(m)  # gradient via adjoint method
```

## API Reference

### Core Functions

#### `construct_objective(forward_op, data_loss, regularization, forward_solver)`

Construct the objective function J(m) = D(u) + R(m).

**Parameters:**
- `forward_op`: Function defining F(u, m) = 0
- `data_loss`: Function defining D(u)
- `regularization`: Function defining R(m)
- `forward_solver`: Function defining the forward solver (optional)

**Returns:**
- `objective`: Objective function J(m) who, when differentiated, yields the
  gradient ∇J(m) as computed via adjoint method (using a custom gradient)

## TODO

The following items are planned for future development to improve the library's interface and performance:

### Interface Refactoring

1. **Optimistix Integration for Forward Solvers**
   - Refactor `construct_objective` to accept forward solvers using the standard [Optimistix](https://github.com/patrick-kidger/optimistix) interface
   - This will provide access to robust, well-tested solvers for nonlinear equations and optimization problems
   - Enable users to easily switch between different solver algorithms (Newton, quasi-Newton, etc.)

2. **Lineax Integration for Adjoint Solvers**
   - Integrate [Lineax](https://github.com/patrick-kidger/lineax) for solving the adjoint equation `(∂F/∂u)ᵀp = -∂D/∂u`
   - Provide access to various linear solvers (direct, iterative, preconditioned)
   - Enable efficient handling of large-scale adjoint systems

### Jacobian Computation Optimization

3. **JVP/VJP Usage**
   - Review and optimize Jacobian computations in `objective_bwd`
   - Use `jax.jvp` (Jacobian-vector products) and `jax.vjp` (vector-Jacobian products) as appropriate
   - Ensure memory-efficient computation of `∂F/∂u` and `∂F/∂m` without explicitly forming full Jacobian matrices when possible

4. **Performance Improvements**
   - Implement efficient checkpointing strategies for memory-limited problems
   - Add support for matrix-free adjoint solvers
   - Optimize for large-scale PDE-constrained optimization problems

### API Enhancements

5. **Solver Configuration**
   - Provide cleaner interfaces for configuring solver tolerances, maximum iterations, and other parameters
   - Add solver diagnostics and convergence monitoring
   - Support for warm starting both forward and adjoint solvers

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use `adjointx` in your research, please cite:

```bibtex
@software{adjointx,
  title={adjointx: Efficient Adjoint Method Implementation in JAX},
  author={Your Name},
  year={2024},
  url={https://github.com/username/adjointx}
}
```
