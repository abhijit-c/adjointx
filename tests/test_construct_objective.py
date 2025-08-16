"""
Tests for adjointx package
"""

import pytest
import jax.numpy as jnp
import jax
import time
from jax.test_util import check_grads
from adjointx import construct_objective


def simple_forward_solver(forward_operator, m):
    """Simple Newton solver for testing"""
    def solve_system(u):
        return forward_operator(u, m)

    # Initial guess
    u = jnp.zeros_like(m)

    # Simple Newton iterations (fixed number for JIT compatibility)
    for _ in range(10):
        residual = solve_system(u)
        jacobian = jax.jacfwd(solve_system)(u)
        u = u - jnp.linalg.solve(jacobian, residual)

    return u


@pytest.fixture
def linear_system_setup():
    """Fixture for linear system test setup"""
    # Define forward problem: Au = m where A is 2x2 matrix
    def forward_operator(u, m):
        A = jnp.array([[2.0, 1.0], [1.0, 2.0]])
        return A @ u - m

    # Data loss: ||u - u_obs||^2
    def data_loss(u):
        u_obs = jnp.array([1.0, 2.0])
        return 0.5 * jnp.sum((u - u_obs)**2)

    # Regularization: ||m||^2
    def regularization(m):
        return 0.01 * jnp.sum(m**2)

    return forward_operator, data_loss, regularization


def test_construct_objective_basic_functionality(linear_system_setup):
    """Test that construct_objective returns a callable function"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    assert callable(objective)


def test_construct_objective_computes_finite_value(linear_system_setup):
    """Test that objective function computes finite values"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    m = jnp.array([3.0, 4.0])
    J = objective(m)

    assert jnp.isfinite(J)
    assert J > 0


def test_construct_objective_gradient_works(linear_system_setup):
    """Test that gradient computation works via adjoint method"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    m = jnp.array([3.0, 4.0])
    grad_J = jax.grad(objective)(m)

    assert grad_J.shape == m.shape
    assert jnp.all(jnp.isfinite(grad_J))


def test_construct_objective_gradient_correctness(linear_system_setup):
    """Test gradient correctness using jax.test_util.check_grads"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    # Test at multiple points to ensure gradient correctness
    test_points = [
        jnp.array([1.0, 2.0]),
        jnp.array([3.0, 4.0]),
        jnp.array([-1.0, 0.5]),
        jnp.array([0.1, -0.3])
    ]

    for m in test_points:
        # check_grads compares analytical gradients with numerical approximations
        # Only use reverse mode since custom_vjp doesn't support forward mode
        # Use relaxed tolerances due to the complexity of the adjoint method computation
        check_grads(objective, (m,), order=1, modes=['rev'], eps=1e-4, rtol=1e-2, atol=1e-2)


def test_gradient_correctness_parametrized(linear_system_setup):
    """Parametrized test for gradient correctness at different points"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    # Test points that should work well for finite differences
    m = jnp.array([2.0, 1.5])
    
    # Use check_grads with more relaxed tolerances for complex adjoint computation
    check_grads(objective, (m,), order=1, modes=['rev'], eps=1e-4, rtol=1e-2, atol=1e-2)


def test_construct_objective_gradient_shape_consistency(linear_system_setup):
    """Test gradient shape consistency for different parameter sizes"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    # Test with original 2D parameters
    m1 = jnp.array([1.0, 2.0])
    grad1 = jax.grad(objective)(m1)
    assert grad1.shape == (2,)

    # Test with different values
    m2 = jnp.array([5.0, -1.0])
    grad2 = jax.grad(objective)(m2)
    assert grad2.shape == (2,)


def test_objective_value_changes_with_parameters(linear_system_setup):
    """Test that objective value changes when parameters change"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    m1 = jnp.array([1.0, 1.0])
    m2 = jnp.array([5.0, 5.0])

    J1 = objective(m1)
    J2 = objective(m2)

    # Objective values should be different for different parameters
    assert not jnp.allclose(J1, J2)


@pytest.mark.parametrize("m_values", [
    jnp.array([0.1, 0.1]),  # Avoid zero which can cause numerical issues
    jnp.array([1.0, 2.0]),
    jnp.array([-1.0, 3.0]),
    jnp.array([2.0, -1.5])  # Reduced magnitude for better numerical stability
])
def test_construct_objective_parametrized(linear_system_setup, m_values):
    """Parametrized test for different parameter values"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    # Test objective evaluation
    J = objective(m_values)
    assert jnp.isfinite(J)

    # Test gradient computation
    grad_J = jax.grad(objective)(m_values)
    assert grad_J.shape == m_values.shape
    assert jnp.all(jnp.isfinite(grad_J))

    # Test gradient correctness using numerical verification
    check_grads(objective, (m_values,), order=1, modes=['rev'], eps=1e-4, rtol=1e-2, atol=1e-2)


def test_adjoint_gradient_no_solver_error():
    """Test that adjoint_gradient raises error when no solver provided"""
    from adjointx import adjoint_gradient

    def dummy_forward_op(u, m):
        return u - m

    def dummy_data_loss(u):
        return jnp.sum(u**2)

    def dummy_regularization(m):
        return 0.01 * jnp.sum(m**2)

    m = jnp.array([1.0, 2.0])

    with pytest.raises(NotImplementedError, match="A forward solver must be provided"):
        adjoint_gradient(dummy_forward_op, dummy_data_loss, dummy_regularization, m)


def test_solve_forward_problem_not_implemented():
    """Test that solve_forward_problem raises NotImplementedError"""
    from adjointx import solve_forward_problem

    def dummy_forward_op(u, m):
        return u - m

    m = jnp.array([1.0, 2.0])

    with pytest.raises(NotImplementedError, match="Forward solver 'newton' not yet implemented"):
        solve_forward_problem(dummy_forward_op, m)


def test_solve_adjoint_equation_not_implemented():
    """Test that solve_adjoint_equation raises NotImplementedError"""
    from adjointx import solve_adjoint_equation

    def dummy_adjoint_op(x):
        return x

    rhs = jnp.array([1.0, 2.0])

    with pytest.raises(NotImplementedError, match="Adjoint equation solver not yet implemented"):
        solve_adjoint_equation(dummy_adjoint_op, rhs)


def test_jit_compilation_performance_benchmark(linear_system_setup):
    """Test that JIT compilation provides significant performance improvement"""
    forward_operator, data_loss, regularization = linear_system_setup

    objective = construct_objective(
        forward_operator,
        data_loss,
        regularization,
        simple_forward_solver
    )

    m = jnp.array([3.0, 4.0])

    # First call - triggers compilation
    start = time.time()
    J1 = objective(m)
    grad1 = jax.grad(objective)(m)
    first_call_time = time.time() - start

    # Second call - should be much faster (already compiled)
    start = time.time()
    J2 = objective(m)
    grad2 = jax.grad(objective)(m)
    second_call_time = time.time() - start

    # Verify results are consistent
    assert jnp.allclose(J1, J2)
    assert jnp.allclose(grad1, grad2)

    # Verify JIT compilation provides speedup
    # Second call should be at least 5x faster than first call
    speedup = first_call_time / second_call_time
    assert speedup >= 5.0, f"Expected speedup >= 5x, got {speedup:.2f}x"

    # Print benchmark results for visibility
    print(f"\nJIT Compilation Benchmark:")
    print(f"First call (with compilation): {first_call_time:.4f}s")
    print(f"Second call (compiled): {second_call_time:.4f}s")
    print(f"Speedup: {speedup:.1f}x")
