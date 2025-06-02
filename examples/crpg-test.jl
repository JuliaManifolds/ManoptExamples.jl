using BenchmarkTools
using Plots; pyplot()
using LaTeXStrings
using Random, LinearAlgebra
using ManifoldDiff, Manifolds, Manopt

"""
    Test Theorem C.1 of the RPGM paper on the SPD manifold
    
    This script implements a test for the convergence rates of the Riemannian
    Proximal Gradient Method (RPGM) as described in Theorem C.1.
    
    We use a function g that is geodesically convex but not strongly geodesically convex.
"""

# Set random seed for reproducibility
Random.seed!(42)

# Define the manifold: Symmetric Positive Definite matrices
n = 3  # dimension
M = SymmetricPositiveDefinite(n)

# Reference point for h function
q₂ = rand(M) # exp(M, Matrix{Float64}(I, n, n), 0.5 * (rand(n, n) + rand(n, n)'))

# Create a fixed matrix A for the linear function
A = (rand(n, n) + rand(n, n)')/2  # symmetric random matrix

# Define objective functions
# g(p) = tr(Ap) - a linear function - geodesically convex but not strongly geodesically convex
function g(M, p)
    return tr(A * p)
end

# Gradient of g - for linear function, it's just A
function grad_g(M, p)
    # The gradient of tr(Ap) with respect to p is A
    # But since we're on a manifold, we need to project A onto the tangent space at p
    return symmetric_projection_on_tangent(M, p, A)
end

# Helper to calculate the riemannian gradient of tr(Ap)
function symmetric_projection_on_tangent(M, p, X)
    # For SPD with affine metric, project X onto tangent space at p
    return p * X * p
end

# Define h
# h(p) = d(p, q₂) - geodesically convex
α = 0.1  # Weight parameter
h(M, p) = α * distance(M, p, q₂)

# Combined objective
f(M, p) = g(M, p) + h(M, p)

# Proximal operator for h (distance function)
function prox_h(M, λ, p)
    return ManifoldDiff.prox_distance(M, α * λ, q₂, p, 1)
end

# Function to estimate smoothness constant for g
function estimate_smoothness_constant(M, g, grad_g, p₀; num_samples=10, step_size=0.1)
    L_estimates = []
    
    for _ in 1:num_samples
        # Generate random direction
        X = randn(n, n)
        X = (X + X')/2  # Make symmetric
        X = X / norm(X)
        
        # Create two points along this direction
        p₁ = p₀
        p₂ = exp(M, p₁, step_size * X)
        
        # Estimate Lipschitz constant of gradient
        grad_diff_norm = norm(parallel_transport_to(M, p₂, grad_g(M, p₂), p₁) - grad_g(M, p₁))
        dist = distance(M, p₁, p₂)
        
        push!(L_estimates, grad_diff_norm / dist)
    end
    
    # Return a conservative estimate (slightly larger than the max)
    return 1.2 * maximum(L_estimates)
end

# Calculate theoretical bounds from Theorem C.1
function calculate_theoretical_bounds(M, f, p_initial, p_optimal, costs, iters, L_g)
    Δ₀ = costs[1] - f(M, p_optimal)
    R = distance(M, p_initial, p_optimal)
    
    # For Hadamard manifolds
    D = 2 * R  # Very conservative estimate

    k_min = -0.5
    ζ_value = √-k_min * D * coth(√-k_min * D)
    
    # Case 1: Fast rate - O(1/2^k)
    bound_fast = [Δ₀ / (2^k) for k in iters]
    
    # Case 2: Slower rate - O(1/k) - with β=1 for constant step size
    β = 1.0
    bound_slower = [Δ₀ * (2*L_g*ζ_value*R^2) / (2*L_g*ζ_value*R^2 + k*β*Δ₀) for k in iters]
    
    return bound_fast, bound_slower
end

# Create initial point
p_initial = rand(M) #Matrix{Float64}(I, n, n)

# First estimate smoothness constant
L_g = estimate_smoothness_constant(M, g, grad_g, p_initial)
println("Estimated L_g: $L_g")

# Find approximate optimal solution with many iterations
println("Finding approximate optimal solution...")
optimal_result = proximal_gradient_method(
    M, 
    (M, p) -> f(M, p), 
    (M, p) -> g(M, p), 
    grad_g, 
    prox_h, 
    p_initial, 
    stepsize = ProxGradBacktracking(; strategy=:convex, initial_stepsize=1.0/L_g),
    stopping_criterion = StopWhenGradientMappingNormLess(1e-12) | StopAfterIteration(5000),
    record = [:Iteration, :Cost, :Iterate],
    return_state = true
)

p_optimal = get_iterate(optimal_result)
optimal_costs = [record[2] for record in get_record(optimal_result)]
f_optimal = minimum(optimal_costs)
println("Approximate optimal value: $f_optimal")

# Run RPGM with constant step size
println("Running RPGM with constant step size...")
const_result = proximal_gradient_method(
    M,
    (M, p) -> f(M, p),
    (M, p) -> g(M, p),
    grad_g,
    prox_h,
    p_initial,
    stepsize = ConstantLength(1.0/L_g),  # Fixed stepsize 1/L_g
    stopping_criterion = StopWhenGradientMappingNormLess(1e-8) | StopAfterIteration(100),
    record = [:Iteration, :Cost, :Iterate],
    return_state = true
)

# Run RPGM with backtracking
println("Running RPGM with backtracking...")
bt_result = proximal_gradient_method(
    M,
    (M, p) -> f(M, p),
    (M, p) -> g(M, p),
    grad_g,
    prox_h,
    p_initial,
    stepsize = ProxGradBacktracking(; strategy=:convex, initial_stepsize=2.0/L_g),
    stopping_criterion = StopWhenGradientMappingNormLess(1e-8) | StopAfterIteration(100),
    record = [:Iteration, :Cost, :Iterate],
    return_state = true
)

# Extract costs and iterations
costs_constant = [record[2] for record in get_record(const_result)]
iters_constant = [record[1] for record in get_record(const_result)]

costs_bt = [record[2] for record in get_record(bt_result)]
iters_bt = [record[1] for record in get_record(bt_result)]

# Make sure we have data to plot
if length(iters_constant) == 0 || length(iters_bt) == 0
    error("No iterations recorded in one of the optimization methods")
end

# Calculate optimality gaps
Δ_constant = [max(cost - f_optimal, 1e-16) for cost in costs_constant]  # Ensure positive values
Δ_bt = [max(cost - f_optimal, 1e-16) for cost in costs_bt]  # Ensure positive values

# Calculate theoretical bounds
bound_fast_constant, bound_slower_constant = calculate_theoretical_bounds(
    M, (M, p) -> f(M, p), p_initial, p_optimal, costs_constant, iters_constant, L_g
)

bound_fast_bt, bound_slower_bt = calculate_theoretical_bounds(
    M, (M, p) -> f(M, p), p_initial, p_optimal, costs_bt, iters_bt, L_g
)

# Reference rate for convergence
ref_rate_linear = [Δ_constant[1]/max(k, 1) for k in iters_constant]  # Avoid division by zero
ref_rate_sqrt = [Δ_constant[1]/sqrt(max(k, 1)) for k in iters_constant]  # Avoid division by zero

# Ensure no NaN or Inf values
has_valid_data = all(isfinite, Δ_constant) && all(isfinite, Δ_bt) && 
                 all(isfinite, bound_fast_constant) && all(isfinite, bound_slower_constant) &&
                 all(isfinite, bound_fast_bt) && all(isfinite, bound_slower_bt)

# if !has_valid_data
#     error("Non-finite values detected in the data")
# end

# Plotting - each plot separately
p1 = plot(
    iters_constant, 
    Δ_constant, 
    yscale=:log10, 
    xscale=:log10,
    label="RPGM Constant",
    xlabel="Iteration", 
    ylabel="Δ (log scale)",
    title="Convergence: Constant Step Size",
    linewidth=2,
    legend=:topright,
    size=(500, 400)  # Explicit size
)

# plot!(p1, iters_constant, bound_fast_constant, 
    #   label="Fast Rate O(1/2^k)", linestyle=:dash)
plot!(p1, iters_constant, bound_slower_constant, 
      label="Slower Rate O(1/k)", linestyle=:dot)
plot!(p1, iters_constant, ref_rate_linear, 
      label="O(1/k) reference", linestyle=:dashdot, linecolor=:black)
plot!(p1, iters_constant, ref_rate_sqrt, 
      label="O(1/√k) reference", linestyle=:dashdotdot, linecolor=:black)

display(p1)  # Show the first plot

p2 = plot(
    iters_bt, 
    Δ_bt, 
    yscale=:log10, 
    label="RPGM Backtracking",
    xlabel="Iteration", 
    ylabel="Δ (log scale)",
    title="Convergence: Backtracking",
    linewidth=2,
    legend=:topright,
    size=(500, 400)  # Explicit size
)
# plot!(p2, iters_bt, bound_fast_bt, label="Fast Rate O(1/2^k)", linestyle=:dash)
plot!(p2, iters_bt, bound_slower_bt, label="Slower Rate O(1/k)", linestyle=:dot)

display(p2)  # Show the second plot

# Only try the combined plot if we have valid data
try
    p_combined = plot(p1, p2, layout=(1, 2), size=(1000, 400))
    savefig(p_combined, "theorem_c1_convergence.png")
    display(p_combined)
catch e
    println("Could not create combined plot: $e")
    println("Individual plots were created successfully.")
end

# Analysis of which bound is tighter
println("\nAnalysis of optimization:")
println("Final optimality gap (constant): $(Δ_constant[end])")
println("Final optimality gap (backtracking): $(Δ_bt[end])")
println("Theoretical rates for first and last iterations:")
# println("Fast rate (1/2^k): First = $(bound_fast_constant[1]), Last = $(bound_fast_constant[end])")
println("Slow rate (1/k): First = $(bound_slower_constant[1]), Last = $(bound_slower_constant[end])")
println("Actual rate: First = $(Δ_constant[1]), Last = $(Δ_constant[end])")

# Analysis of optimization condition
function analyze_bound_conditions(M, p_optimal, Δs, L_g, iterates)
    conditions = []
    
    for (i, (Δ, p)) in enumerate(zip(Δs, iterates))
        # Calculate parameters
        k_min = -0.5
        D = 2*distance(M, p, p_optimal)
        ζ_value = √-k_min * D * coth(√-k_min * D)
        λ = 1/L_g      # Constant step size
        d_squared = D^2
        
        # Check condition: λΔ/(ζd²) ≥ 1
        condition_value = (λ * Δ) / (ζ_value * d_squared)
        push!(conditions, condition_value)
    end
    
    return conditions
end

# Get iterates
iterates_constant_objs = [record[3] for record in get_record(const_result)]

# Calculate condition values
conditions = analyze_bound_conditions(
    M, p_optimal, Δ_constant, L_g, iterates_constant_objs
)

# Plot the condition values
if !isempty(conditions) && all(isfinite, conditions)
    cond_plot = plot(
        1:length(conditions), 
        conditions, 
        label="λΔ/(ζd²)",
        xlabel="Iteration", 
        ylabel="Condition Value",
        title="Theorem C.1 Condition Analysis",
        linewidth=2,
        size=(600, 400)
    )
    hline!(cond_plot, [1.0], label="Threshold", linestyle=:dash)
    display(cond_plot)
    savefig(cond_plot, "theorem_c1_condition.png")
end