# adjointx

A JAX module for computing gradients through functionals constrained by equations using the adjoint method.

## Overview

`adjointx` provides efficient gradient computation for optimization problems of the form:

$$\min_m J(m) = \ell(u(m), m) + R(m)$$

subject to the constraint:

$$F(u(m), m) = 0$$

where:
- $m$ is the control/parameter vector
- $u(m)$ is the state variable that depends on $m$ through the constraint
- $\ell(u, m)$ is the data loss term
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

$$\mathcal{L}(u, m, p) = \ell(u, m) + R(m) + p^T F(u, m)$$

The adjoint method exploits the fact that at the optimum, the constraint is satisfied ($F(u, m) = 0$), so:

$$\frac{dJ}{dm} = \frac{\partial \mathcal{L}}{\partial m}$$

The adjoint variable $p$ must be selected to satisfy the **adjoint equation**:

$$\frac{\partial \mathcal{L}}{\partial u} = 0 \implies \left(\frac{\partial F}{\partial u}\right)^T p = -\frac{\partial \ell}{\partial u}$$

Once $p$ is computed, the gradient is given by:

$$\frac{dJ}{dm} = \frac{\partial \ell}{\partial m} + \frac{\partial R}{\partial m} + p^T \frac{\partial F}{\partial m}$$

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
uv add -e .

## Quick Start

```python
import jax.numpy as jnp
from adjointx import adjoint_gradient

# Define your forward problem F(u, m) = 0
def forward_operator(u, m):
    """Example: simple linear system Au = m"""
    A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
    return A @ u - m

# Define your objective function ℓ(u, m)
def objective(u, m):
    """Example: least squares data fitting"""
    u_obs = jnp.array([1.0, 2.0])  # observed data
    return 0.5 * jnp.sum((u - u_obs)**2)

# Define regularization R(m)
def regularization(m):
    """Example: L2 regularization"""
    return 0.01 * jnp.sum(m**2)

# Compute gradient using adjoint method
m = jnp.array([3.0, 4.0])  # initial parameters
grad_m = adjoint_gradient(
    forward_operator,
    objective,
    regularization,
    m
)
```

## Examples

### Example 1: Parameter Estimation in ODEs

```python
import jax
import jax.numpy as jnp
from adjointx import solve_adjoint_system

def ode_residual(u, m, t):
    """du/dt = -m[0] * u + m[1], with initial condition u(0) = u0"""
    dudt = -m[0] * u + m[1]
    # Residual for time stepping scheme
    return u[1:] - u[:-1] - dt * dudt[:-1]

def data_fit(u, m):
    """Fit to observed trajectory"""
    u_obs = jnp.array([...])  # your observed data
    return 0.5 * jnp.sum((u - u_obs)**2)

# Use adjoint method to find parameters
m_optimal = optimize_with_adjoint(ode_residual, data_fit, m_init)
```

### Example 2: PDE-Constrained Optimization

```python
def heat_equation_residual(u, m):
    """2D heat equation with parameter-dependent source term"""
    # Discretized ∇²u - m*f = 0
    laplacian_u = finite_difference_laplacian(u)
    source = m * source_pattern
    return laplacian_u - source

def boundary_objective(u, m):
    """Minimize temperature at specific boundary points"""
    boundary_temps = extract_boundary(u)
    target_temps = jnp.array([...])
    return jnp.sum((boundary_temps - target_temps)**2)

# Optimize heat source distribution
optimal_source = adjoint_optimize(
    heat_equation_residual,
    boundary_objective,
    initial_guess
)
```

## API Reference

### Core Functions

#### `adjoint_gradient(forward_op, objective, regularization, params)`

Compute gradients using the adjoint method.

**Parameters:**
- `forward_op`: Function defining F(u, m) = 0
- `objective`: Function defining ℓ(u, m)
- `regularization`: Function defining R(m)
- `params`: Current parameter vector m

**Returns:**
- `gradient`: ∇J(m) computed via adjoint method

#### `solve_forward_problem(forward_op, params, solver='newton')`

Solve the forward problem F(u, m) = 0 for given parameters.

**Parameters:**
- `forward_op`: Forward operator function
- `params`: Parameter vector
- `solver`: Solver method ('newton', 'bicgstab', 'gmres')

**Returns:**
- `state`: Solution u(m)

#### `solve_adjoint_equation(adjoint_op, rhs)`

Solve the adjoint equation (∂F/∂u)ᵀp = rhs.

**Parameters:**
- `adjoint_op`: Adjoint operator (∂F/∂u)ᵀ
- `rhs`: Right-hand side vector

**Returns:**
- `adjoint`: Adjoint variable p

## Advanced Usage

### Custom Solvers

```python
from adjointx.solvers import NewtonSolver, LinearSolver

# Configure custom Newton solver for forward problem
newton_solver = NewtonSolver(
    max_iterations=50,
    tolerance=1e-12,
    line_search=True
)

# Configure iterative solver for adjoint system
adjoint_solver = LinearSolver(
    method='bicgstab',
    preconditioner='ilu',
    tolerance=1e-10
)

gradient = adjoint_gradient(
    forward_op,
    objective,
    regularization,
    params,
    forward_solver=newton_solver,
    adjoint_solver=adjoint_solver
)
```

### Checkpointing for Memory Efficiency

```python
from adjointx.checkpointing import checkpoint_adjoint

# Use checkpointing for large problems
gradient = checkpoint_adjoint(
    forward_op,
    objective,
    params,
    num_checkpoints=10  # Trade computation for memory
)
```

## Performance Tips

1. **Use JAX transformations**: `jit`, `vmap`, and `pmap` work seamlessly with adjoint computations
2. **Efficient linear solvers**: Choose appropriate solvers for your problem structure
3. **Checkpointing**: For memory-constrained problems, use gradient checkpointing
4. **Preconditioning**: Implement problem-specific preconditioners for faster convergence

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

## References

- Plessix, R. E. (2006). A review of the adjoint-state method for computing the gradient of a functional with geophysical applications. *Geophysical Journal International*, 167(2), 495-503.
- Hinze, M., Pinnau, R., Ulbrich, M., & Ulbrich, S. (2008). *Optimization with PDE constraints* (Vol. 23). Springer Science & Business Media.
- Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., ... & Wanderman-Milne, S. (2018). JAX: composable transformations of Python+ NumPy programs.
