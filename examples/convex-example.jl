### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 2ba07436-1910-11f0-101e-9dc5d467d3df
begin
	using Pkg; Pkg.activate(".")
	using Manifolds, Manopt, ManifoldDiff
	using FiniteDifferences, Random
	using Plots
	using ColorSchemes, PlutoUI, LaTeXStrings
end

# ╔═╡ 285abba2-0bad-45e9-acfb-84b83b8b68eb
begin
	Random.seed!(42)

	result_color = RGBA{Float64}(colorant"#EE7733") # Tol Vibrant Orange
	
	M = SymmetricPositiveDefinite(3)
	q1 = rand(M)
	
	X1 = rand(M; vector_at=q1)
	X1 /= norm(M, q1, X1)
	q2 = exp(M, q1, 2X1)

	X2 = rand(M; vector_at=q2)
	X2 /=norm(M, q2, X2)
	p0 = exp(M, q2, X2)#rand(M)
	
	r = distance(M, q1, q2)

	τ = 1.5

	γ(t) = geodesic(M, q1, log(M, q1, q2), t)

	p_star = τ > 1 ? q1 : (τ < 1 ? q2 : γ(rand()))

	#D = max(distance(M, p0, q1), distance(M, p0, q2), r, distance(M, p0, p_star))
	D = 2maximum([distance(M, p0, a) for a in [q1, q2, p_star]])
end

# ╔═╡ 5987498a-b1db-44ca-a181-a1afe4a67d7a
function is_close(M, p, q, dist)
	return distance(M, p, q) ≤ dist
end	

# ╔═╡ 1d930f64-4455-4f99-9956-4bab7991ceb8
[is_close(M, p0, q, D) for q in [p0, q1, q2, p_star]]

# ╔═╡ 25dca9df-7df3-4e5a-9f4b-2b9404073173
if split("$M", "(")[1] == "Sphere"
	k_min = 1.0
	k_max = 1.0
elseif split("$M", "(")[1] == "Hyperbolic"
	k_min = -1.0
	k_max = -1.0
elseif split("$M", "(")[1] == "SymmetricPositiveDefinite"
	k_min = -0.5
	k_max = 0.0
end

# ╔═╡ e5467f25-be10-437d-91de-82fed6054fdc
begin
function sectional_curvature(M, p)
    X = rand(M; vector_at=p)
    Y = rand(M; vector_at=p)
    Y = Y - inner(M, p, X, Y) / norm(M, p, X)^2 * X
    R = riemann_tensor(M, p, X, Y, Y)
    return inner(M, p, R, X) / (norm(M, p, X)^2 * norm(M, p, Y)^2 - inner(M, p, X, Y)^2)
end
function ζ_1(k_min, dist)
    (k_min < zero(k_min)) && return sqrt(-k_min) * dist * coth(sqrt(-k_min) * dist)
    (k_min ≥ zero(k_min)) && return one(k_min)
end
function ζ_2(k_max, dist)
    (k_max ≤ zero(k_max)) && return one(k_max)
    (k_max > zero(k_max)) && return sqrt(k_max) * dist * cot(sqrt(k_max) * dist)
end
ϱ(k_min, k_max, dist) = max(ζ_1(k_min, dist) - 1, 1- (ζ_2(k_max, dist)))
ζ(k_min, k_max, dist) = max(ζ_1(k_min, dist), - (ζ_2(k_max, dist)))
end

# ╔═╡ 6af12b18-eab1-4d6f-8ade-de35f40ae33b
number_eltype(D)

# ╔═╡ e8bc7dee-6668-41b8-8ab9-0b045189ab8c
ζ_D = ζ_1(k_min, D)

# ╔═╡ a424f635-cdbe-4f8e-b126-540426567b15
1/ζ_D

# ╔═╡ 30c67be0-ae97-4336-90d0-bbff2c3c7a9c
begin
	h(M, p) = τ * distance(M, p, q1)
	prox_h(M, λ, p) = ManifoldDiff.prox_distance(M, τ*λ, q1, p, 1)
	
	g(M, p) = distance(M, p, q2)
	grad_g(M, p) = ManifoldDiff.grad_distance(M, q2, p, 1)
	
	f(M, p) = g(M, p) + h(M, p)
	proxes = Function[(M, λ, p) -> ManifoldDiff.prox_distance(M, λ, q2, p, 1)]
	push!(proxes, (M, λ, p) -> prox_h(M, λ, p))

	λ_min = 1/g(M, p_star)
	λ_max = coth(g(M, p_star))
	λ_r = 2*g(M, p_star)/(g(M, p_star)*coth(g(M, p_star)) + 1)
end

# ╔═╡ d5ad4577-8c20-44db-b1a9-5f4617cf9ef3
# Function to calculate the inequality value
function calculate_inequality_value(λ, gradient_norm, ω, λ_max, λ_min)
    if λ == 0 || gradient_norm == 0 || ω == 0
        return 0.0
    end
    
    argument = λ * gradient_norm * sqrt(-ω)
    
    # For small arguments, use Taylor expansion to avoid precision issues
    sinc = if abs(argument) < 1e-10
        1.0
    else
        sinh(argument) / argument
    end
    
    term1 = max(abs(1 - λ_max * λ), abs(1 - λ_min * λ))
    
    return sinc * term1
end

# ╔═╡ 2d9938a9-efbc-4f76-afb7-ad0fae101645
begin
    # Generate data points for the plot
    λ_values = range(1e-3, max(r/τ, 1/λ_min, 2/λ_max)+1e-3, length=200)
    inequality_values = [calculate_inequality_value(λ, 1.0, k_min, λ_max, λ_min) for λ in λ_values]
    
    # Plot the inequality
    plot(λ_values, inequality_values, 
        label="Inequality Value",
        xlabel=L"\lambda",
        ylabel="Value",
        title="Inequality Analysis",
        lw=2,
        legend=:topright,
        size=(800, 500),
        grid=true,
        framestyle=:box
    )
    
    # Add a horizontal line at y=1 to show the boundary
    hline!([1.0], label="Inequality Boundary", ls=:dash, lw=2, c=:red)
    
    # Add shaded region for where inequality is satisfied
    satisfied_indices = findall(v -> v < 1.0, inequality_values)
    if !isempty(satisfied_indices)
        first_idx = satisfied_indices[1]
        last_idx = satisfied_indices[end]
        if last_idx - first_idx > 0  # Ensure there's actually a region to shade
            plot!(λ_values[first_idx:last_idx], inequality_values[first_idx:last_idx], 
                  fillrange=1, fillalpha=0.2, c=:green, label="Satisfied Region")
        end
    end
    
    # Add vertical lines at λ = 1/Λᵐᵃˣ and λ = 1/Λᵐⁱⁿ
    if λ_max > 0
        vline!([1/λ_max], label=L"\lambda = 1/\Lambda^{max}", ls=:dot, lw=1.5, c=:blue)
		vline!([2/λ_max], label=L"\lambda = 2/\Lambda^{max}", ls=:dot, lw=1.5, c=:black)
    end
    if λ_min > 0
        vline!([1/λ_min], label=L"\lambda = 1/\Lambda^{min}", ls=:dot, lw=1.5, c=:purple)
    end
	vline!([λ_r], label=L"\lambda = 2r/(rcoth(r)+1)", ls=:dot, lw=1.5, c=:green)
	vline!([r/τ], label=L"\lambda = r/τ", ls=:dot, lw=1.5, c=:orange)
	vline!([2/ζ_D], label=L"\lambda = 1/ζ_D", ls=:dot, lw=1.5, c=:red)
    
    # Enforce y-axis limits 
    plot!(ylim=(0, max(1.5, maximum(inequality_values) * 1.1)))
end

# ╔═╡ 2b4ec87d-9d2e-498f-b89e-86960ffd07ca
pg = proximal_gradient_method(
	M, 
	f, 
	g, 
	grad_g, 
	prox_h, 
	p0;
	stepsize=
		ConstantLength(1/ζ_D),
		#ProxGradBacktracking(; strategy=:convex, initial_stepsize=2/λ_max),#r/τ-1e-1),
	debug=[
		:Iteration, 
		(:Cost, "F(p): %1.8f "),
		" ", 
		(:Change, " last change: %1.8f "),
		(:last_stepsize, " last step size: %1.8f "),
		1, 
		"\n", 
		:Stop,
	],
	record=[:Iteration, :Cost, :Iterate],
	return_state=true,
	stopping_criterion=StopWhenGradientMappingNormLess(1e-8) |
					   StopAfterIteration(5000) | StopWhenCostNaN(),
)

# ╔═╡ 6f9c197c-3a38-43a6-aede-4c488c708063
begin
pg_result = get_solver_result(pg)
pg_record = get_record(pg)
# Calculate the minimum cost for relative error
min_cost_pg = minimum(record[2] for record in pg_record)
iterations = [record[1] for record in pg_record]
relative_errors_pg = [max(record[2] - min_cost_pg, 1e-16) for record in pg_record]

# Get initial error for scaling reference lines
initial_error_pg = relative_errors_pg[1]

# Create reference trajectories
# O(1/√k)
ref_rate_sqrt_pg = [initial_error_pg/√k for k in iterations]
# O(1/k)
ref_rate_1_pg = [initial_error_pg/k for k in iterations]
# O(1/k²)
ref_rate_2_pg = [initial_error_pg/k^2 for k in iterations]
# O(1/2^k)
ref_rate_2k_pg = [initial_error_pg/2^k for k in iterations]

# Create the convergence plot
convergence_plot_pg = plot(
    iterations,
    relative_errors_pg;
    xscale=:log10,
    yscale=:log10,
    xlabel="Iteration (k)",
    ylabel="f(xₖ) - f*",
    label="Proximal Gradient",
    linewidth=2,
    color=result_color,
    marker=:none,
    grid=true,
    legend=:bottomleft
)

# Add reference lines
#=plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_sqrt_pg;
    linestyle=:dash,
    linewidth=1.5,
    color=:black,
    label="O(1/√k)"
)
=#
plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_1_pg;
    linestyle=:dashdot,
    linewidth=1.5,
    color=:black,
    label="O(1/k)"
)
plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_2_pg;
    linestyle=:dot,
    linewidth=1.5,
    color=:black,
    label="O(1/k²)"
)
plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_2k_pg;
    linestyle=:solid,
    linewidth=1.5,
    color=:black,
    label="O(1/2^k)"
)
end

# ╔═╡ 520c8dbc-174e-4f98-8bcb-254a7735aaf5
all(x -> x == true, [distance(M, p0, s[3]) ≤ D for s in pg_record])

# ╔═╡ 93e12c6f-c331-451c-b775-f705dc23e1f4
pg_result ≈ p_star

# ╔═╡ bf769772-c52b-434f-a1db-ecaf207b5ca4
begin
	distances_pg = [max(distance(M, p_star, record[3]), 1e-16) for record in pg_record]

	# Get initial error for scaling reference lines
	initial_distance_pg = distances_pg[1]
	
	# Create reference trajectories
	# O(1/k)
	dist_ref_rate_1_pg = [initial_distance_pg/k for k in iterations]
	# O(1/k²)
	dist_ref_rate_2_pg = [initial_distance_pg/k^2 for k in iterations]

	# Create the convergence plot
	distances_plot_pg = plot(
	    iterations,
	    distances_pg;
	    xscale=:log10,
	    yscale=:log10,
	    xlabel="Iteration (k)",
	    ylabel="dist(xₖ, x*)",
	    label="Proximal Gradient",
	    linewidth=2,
	    color=result_color,
	    marker=:none,
	    grid=true,
	    legend=:bottomleft
	)
	plot!(
	    distances_plot_pg,
	    iterations,
	    dist_ref_rate_1_pg;
	    linestyle=:dashdot,
	    linewidth=1.5,
	    color=:black,
	    label="O(1/k)"
	)
	plot!(
	    distances_plot_pg,
	    iterations,
	    dist_ref_rate_2_pg;
	    linestyle=:dot,
	    linewidth=1.5,
	    color=:black,
	    label="O(1/k²)"
	)
end

# ╔═╡ Cell order:
# ╠═2ba07436-1910-11f0-101e-9dc5d467d3df
# ╠═285abba2-0bad-45e9-acfb-84b83b8b68eb
# ╠═5987498a-b1db-44ca-a181-a1afe4a67d7a
# ╠═1d930f64-4455-4f99-9956-4bab7991ceb8
# ╠═25dca9df-7df3-4e5a-9f4b-2b9404073173
# ╠═e5467f25-be10-437d-91de-82fed6054fdc
# ╠═6af12b18-eab1-4d6f-8ade-de35f40ae33b
# ╠═e8bc7dee-6668-41b8-8ab9-0b045189ab8c
# ╠═a424f635-cdbe-4f8e-b126-540426567b15
# ╠═30c67be0-ae97-4336-90d0-bbff2c3c7a9c
# ╠═d5ad4577-8c20-44db-b1a9-5f4617cf9ef3
# ╠═2d9938a9-efbc-4f76-afb7-ad0fae101645
# ╠═2b4ec87d-9d2e-498f-b89e-86960ffd07ca
# ╠═520c8dbc-174e-4f98-8bcb-254a7735aaf5
# ╠═93e12c6f-c331-451c-b775-f705dc23e1f4
# ╠═6f9c197c-3a38-43a6-aede-4c488c708063
# ╠═bf769772-c52b-434f-a1db-ecaf207b5ca4
