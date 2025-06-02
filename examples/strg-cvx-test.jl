using Manifolds, Manopt, LinearAlgebra, Random

"""
    check_strong_geodesic_convexity(M, f, grad_f, μ; num_points=20, tol=1e-10)

Check if a function is μ-strongly geodesically convex on a Riemannian manifold by
testing random pairs of points against the strong convexity condition.

Parameters:
- M: A Riemannian manifold
- f: A function f(M, p) -> Real
- grad_f: The gradient of f, grad_f(M, p) -> Vector in the tangent space
- μ: The strong convexity parameter to check
- num_points: Number of random points to sample
- tol: Tolerance for numerical errors

Returns:
- is_convex: Boolean indicating if the function appears to be μ-strongly convex
- min_violation: The minimum violation of the strong convexity inequality
- worst_pair: The pair of points with the minimum violation
"""
function check_strong_geodesic_convexity(M, f, grad_f, μ; num_points=1000, tol=1e-12)
    Random.seed!(42)  # For reproducibility
    
    # Sample points from the manifold
    points = [rand(M) for _ in 1:num_points]
    
    min_violation = Inf
    worst_pair = (nothing, nothing)
    
    for i in 1:num_points
        for j in 1:num_points
            if i == j
                continue
            end
            
            p = points[i]
            q = points[j]
            
            # Calculate the terms for the strong convexity inequality
            f_p = f(M, p)
            f_q = f(M, q)
            grad_p = grad_f(M, p)
            log_p_q = log(M, p, q)
            dist_squared = distance(M, p, q)^2
            
            inner_prod = inner(M, p, grad_p, log_p_q)
            
            # Strong convexity condition: f(q) ≥ f(p) + (grad_f(p), log_p(q)) + (μ/2)d²(p,q)
            # Rearranged as: f(q) - f(p) - (grad_f(p), log_p(q)) - (μ/2)d²(p,q) ≥ 0
            violation = f_q - f_p - inner_prod - (μ / 2) * dist_squared
            
            if violation < min_violation
                min_violation = violation
                worst_pair = (p, q)
            end
        end
    end
    
    is_convex = min_violation >= -tol
    
    return is_convex, min_violation, worst_pair
end

# Example: Test a function on the sphere
function sphere_example()
    # Define the manifold (2-dimensional sphere S²)
    M = Sphere(2)
    
    # Define a function and its gradient
    # Example: squared distance to a reference point outside the sphere
    reference_point = [0.0, 0.0, 2.0]
    
    function example_function(M, p)
        return norm(p - reference_point)^2
    end
    
    function example_gradient(M, p)
        # Project the Euclidean gradient onto the tangent space
        euclidean_grad = 2 * (p - reference_point)
        return project(M, p, euclidean_grad)
    end
    
    # Check for different values of μ
    μ_values = [0.5, 1.0, 1.5, 2.0]
    
    for μ in μ_values
        is_convex, violation, worst_pair = check_strong_geodesic_convexity(
            M, example_function, example_gradient, μ)
        
        if is_convex
            println("For μ = $μ: The function IS strongly geodesically convex (min violation = $violation)")
        else
            println("For μ = $μ: The function is NOT strongly geodesically convex (min violation = $violation)")
            p, q = worst_pair
            println("  Worst violation at points:")
            println("  p = $p")
            println("  q = $q")
        end
    end
end

# Example: Test a function on symmetric positive definite matrices

# Define the manifold: 2×2 symmetric positive definite matrices
M = SymmetricPositiveDefinite(2)

# Define a function
function example_function(M, p)
    return log(det(p))^4
    # return -log(det(p))
end

# Gradient of -log(det(p)) is -p^(-1)
function example_gradient(M, p)
    return 4 * (log(det(p)))^3 * p
    # return -inv(p)
end

function example_hessian(M, p, X)
    return (12 * (log(det(p)))^2) * tr(inv(p) * X) * p
end



# Check for different values of μ
μ_values = [0.0, 1e-8, 1e-4, 1e-2, 1e0, 1e1]

for μ in μ_values
    is_convex, violation, worst_pair = check_strong_geodesic_convexity(
        M, example_function, example_gradient, μ)
    
    if is_convex
        println("For μ = $μ: The function COULD BE strongly geodesically convex (min violation = $violation)")
    else
        println("For μ = $μ: The function is NOT strongly geodesically convex (min violation = $violation)")
        p, q = worst_pair
        println("  Worst violation at points:")
        println("  p = $p")
        println("  q = $q")
    end
end

# Run the examples
# sphere_example()