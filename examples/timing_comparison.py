"""
Timing comparison between naive JAX grad and adjointx package for high-dimensional problems.

This example demonstrates the computational advantages of using the adjoint method
for gradient computation in high-dimensional inverse problems.
"""

import jax
import jax.numpy as jnp
import timeit
from adjointx import construct_objective


def create_high_dim_linear_problem(dim=1024):
    """Create a high-dimensional linear inverse problem."""

    # Create a well-conditioned random matrix A
    key = jax.random.PRNGKey(42)
    A = jax.random.normal(key, (dim, dim))
    # Make it symmetric positive definite for stability
    A = A @ A.T + 0.01 * jnp.eye(dim)

    # Synthetic observed data
    u_true = jnp.ones(dim)
    u_obs = A @ u_true + 0.01 * jax.random.normal(jax.random.PRNGKey(123), (dim,))

    def forward_operator(u, m):
        """Linear forward operator: Au = m"""
        return A @ u - m

    def data_loss(u):
        """Data fitting term: ||u - u_obs||^2"""
        return 0.5 * jnp.sum((u - u_obs) ** 2)

    def regularization(m):
        """L2 regularization: λ||m||^2"""
        return 0.001 * jnp.sum(m**2)

    def linear_solver(forward_op, m):
        """Direct linear solver for Au = m"""
        return jnp.linalg.solve(A, m)

    return forward_operator, data_loss, regularization, linear_solver, u_obs


def naive_objective_function(forward_op, data_loss, regularization, solver):
    """Naive implementation that JAX will differentiate directly."""

    def objective(m):
        u = solver(forward_op, m)
        return data_loss(u) + regularization(m)

    return objective


def benchmark_gradients(dim=1024, n_runs=50):
    """Compare timing between naive JAX grad and adjoint method."""

    # Setup problem
    forward_op, data_loss, reg, solver, _ = create_high_dim_linear_problem(dim)

    # Create test parameter vector
    m_test = jnp.ones(dim)

    # Method 1: Naive JAX automatic differentiation
    naive_obj = naive_objective_function(forward_op, data_loss, reg, solver)
    naive_grad_fn = jax.grad(naive_obj)

    # Method 2: adjointx with custom adjoint implementation
    adjoint_obj = construct_objective(forward_op, data_loss, reg, solver)
    adjoint_grad_fn = jax.grad(adjoint_obj)

    # Compile both functions first (silent compilation)
    _ = naive_grad_fn(m_test)
    _ = adjoint_grad_fn(m_test)

    # Timing function
    def time_function(func, args):
        return timeit.timeit(lambda: func(*args), number=n_runs) / n_runs

    # Benchmark both approaches
    naive_time = time_function(naive_grad_fn, (m_test,))
    adjoint_time = time_function(adjoint_grad_fn, (m_test,))

    # Verify gradients are similar
    naive_grad = naive_grad_fn(m_test)
    adjoint_grad = adjoint_grad_fn(m_test)
    max_diff = jnp.max(jnp.abs(naive_grad - adjoint_grad))
    rel_error = max_diff / jnp.max(jnp.abs(naive_grad))

    # Check gradient verification
    if rel_error > 1e-6:
        print(f"Warning: Large gradient difference for {dim}D (rel_error: {rel_error:.2e})")

    speedup = naive_time / adjoint_time
    return naive_time, adjoint_time, speedup


def main():
    """Run timing comparison for different problem sizes."""

    print("=" * 80)
    print("ADJOINTX TIMING COMPARISON")
    print("=" * 80)
    print("Comparing naive JAX grad vs adjoint method for high-dimensional")
    print("linear inverse problems of the form:")
    print("  min_m ||Au(m) - u_obs||² + λ||m||²")
    print("  subject to: Au(m) = m")
    print("=" * 80)

    # Test different problem dimensions
    dimensions = [128, 256, 512, 1024, 2048, 4096]
    
    # Store results for table output
    results = []
    
    for dim in dimensions:
        print(f"\nBenchmarking {dim}D problem...")
        naive_time, adjoint_time, speedup = benchmark_gradients(dim=dim, n_runs=50)
        results.append((dim, naive_time * 1000, adjoint_time * 1000, speedup))
    
    # Print consolidated results table
    print("\n" + "=" * 80)
    print("CONSOLIDATED RESULTS")
    print("=" * 80)
    print(f"{'Dimension':<12} {'Naive (ms)':<12} {'Adjoint (ms)':<14} {'Speedup':<10}")
    print("-" * 80)
    for dim, naive_ms, adjoint_ms, speedup in results:
        print(f"{dim:<12} {naive_ms:8.2f}    {adjoint_ms:8.2f}      {speedup:5.1f}x")
    print("=" * 80)


if __name__ == "__main__":
    main()

