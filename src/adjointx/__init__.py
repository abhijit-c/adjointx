"""
adjointx: A JAX module for computing gradients through functionals constrained by equations using the adjoint method.
"""

import jax
import jax.numpy as jnp
from jax import custom_vjp
from typing import Callable, Tuple


def construct_objective(
    forward_operator: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    data_loss: Callable[[jnp.ndarray], float],
    regularization: Callable[[jnp.ndarray], float],
    forward_solver: Callable[[Callable, jnp.ndarray], jnp.ndarray]
) -> Callable[[jnp.ndarray], float]:
    """
    Construct an objective function J(m) = D(u(m)) + R(m) with adjoint-based gradients.

    The objective is subject to the constraint F(u(m), m) = 0, where u(m) is found
    by solving the forward problem using the provided forward solver.

    Args:
        forward_operator: Function F(u, m) defining the constraint F(u, m) = 0
        data_loss: Function D(u) computing the data fitting term
        regularization: Function R(m) computing the regularization term
        forward_solver: Function that takes (forward_operator, m) and returns u(m)
                       by solving F(u, m) = 0

    Returns:
        objective: Function that takes m and returns J(m) with adjoint-based gradients
    """

    @custom_vjp
    @jax.jit
    def objective(m: jnp.ndarray) -> float:
        """Compute J(m) = D(u(m)) + R(m)"""
        # Solve forward problem F(u, m) = 0 to get u(m)
        u = forward_solver(forward_operator, m)

        # Compute objective J(m) = D(u(m)) + R(m)
        data_term = data_loss(u)
        reg_term = regularization(m)

        return data_term + reg_term

    @jax.jit
    def objective_fwd(m: jnp.ndarray) -> Tuple[float, Tuple[jnp.ndarray, jnp.ndarray]]:
        """Forward pass: compute objective and save state for backward pass"""
        # Solve forward problem
        u = forward_solver(forward_operator, m)

        # Compute objective components
        data_term = data_loss(u)
        reg_term = regularization(m)

        # Save state needed for backward pass
        residuals = (u, m)

        return data_term + reg_term, residuals

    @jax.jit
    def objective_bwd(residuals: Tuple[jnp.ndarray, jnp.ndarray], cotangent: float) -> Tuple[jnp.ndarray]:
        """Backward pass: compute gradient using adjoint method"""
        u, m = residuals

        # Compute ∂D/∂u
        dD_du = jax.grad(data_loss)(u)

        # Compute ∂R/∂m
        dR_dm = jax.grad(regularization)(m)

        # Compute Jacobian of the forward operator F(u, m) w.r.t. u
        dF_du = jax.jacfwd(forward_operator, argnums=0)(u, m)  # ∂F/∂u

        # Solve adjoint equation: (∂F/∂u)^T p = -∂D/∂u
        # This gives us the adjoint variable p
        rhs = -dD_du
        p = jnp.linalg.solve(dF_du.T, rhs)

        # Compute gradient using adjoint method:
        # dJ/dm = ∂R/∂m + p^T ∂F/∂m
        # Use VJP to compute p^T ∂F/∂m without forming the full Jacobian
        _, vjp_fn = jax.vjp(lambda m_var: forward_operator(u, m_var), m)
        p_T_dF_dm = vjp_fn(p)[0]  # This gives us p^T @ (∂F/∂m)

        dJ_dm = dR_dm + p_T_dF_dm

        return (cotangent * dJ_dm,)

    # Register custom VJP
    objective.defvjp(objective_fwd, objective_bwd)

    return objective


def adjoint_gradient(
    forward_operator: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    data_loss: Callable[[jnp.ndarray], float],
    regularization: Callable[[jnp.ndarray], float],
    params: jnp.ndarray,
    forward_solver: Callable[[Callable, jnp.ndarray], jnp.ndarray] | None = None
) -> jnp.ndarray:
    """
    Compute gradients using the adjoint method.

    Args:
        forward_operator: Function defining F(u, m) = 0
        data_loss: Function defining D(u)
        regularization: Function defining R(m)
        params: Current parameter vector m
        forward_solver: Forward solver function (if None, raises NotImplementedError)

    Returns:
        gradient: ∇J(m) computed via adjoint method
    """
    if forward_solver is None:
        raise NotImplementedError("A forward solver must be provided")

    # Construct objective with adjoint gradients
    objective = construct_objective(forward_operator, data_loss, regularization, forward_solver)

    # Compute gradient
    return jax.grad(objective)(params)


def solve_forward_problem(
    forward_op: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    params: jnp.ndarray,
    solver: str = 'newton'
) -> jnp.ndarray:
    """
    Solve the forward problem F(u, m) = 0 for given parameters.

    Args:
        forward_op: Forward operator function F(u, m)
        params: Parameter vector m
        solver: Solver method ('newton', 'bicgstab', 'gmres')

    Returns:
        state: Solution u(m)
    """
    raise NotImplementedError(f"Forward solver '{solver}' not yet implemented")


def solve_adjoint_equation(
    adjoint_op: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: jnp.ndarray
) -> jnp.ndarray:
    """
    Solve the adjoint equation (∂F/∂u)ᵀp = rhs.

    Args:
        adjoint_op: Adjoint operator (∂F/∂u)ᵀ
        rhs: Right-hand side vector

    Returns:
        adjoint: Adjoint variable p
    """
    raise NotImplementedError("Adjoint equation solver not yet implemented")


# Export public API
__all__ = [
    'construct_objective',
    'adjoint_gradient',
    'solve_forward_problem',
    'solve_adjoint_equation',
]
