using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots#; pgfplotsx()#PGFPlotsX
using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples

function artificial_signal(M, 
    pts::Integer=100; a::Real=0.0, b::Real=1.0, T::Real=(b - a) / 2
)
    t = range(a, b; length=pts)
    x = [[s, sign(sin(2 * π / T * s))] for s in t]
    y = [
        [x[1]]
        [
            x[i] for
            i in 2:(length(x) - 1) if (x[i][2] != x[i + 1][2] || x[i][2] != x[i - 1][2])
        ]
        [x[end]]
    ]
	if M == Hyperbolic(2)
		y = map(z -> Manifolds._hyperbolize(M, z), y)
	end
    data = []
    geodesics = []
    l = Int(round(pts * T / (2 * (b - a))))
    for i in 1:2:(length(y) - 1)
        append!(
            data,
            shortest_geodesic(M, y[i], y[i + 1], range(0.0, 1.0; length=l)),
        )
        if i + 2 ≤ length(y) - 1
            append!(
                geodesics,
                shortest_geodesic(M, y[i], y[i + 1], range(0.0, 1.0; length=l)),
            )
            append!(
                geodesics,
                shortest_geodesic(
                    M, y[i + 1], y[i + 2], range(0.0, 1.0; length=l)
                ),
            )
        end
    end
    return data, geodesics
end

Random.seed!(100)
N = 1000
α = 1/N
σ = 0.1
atol = 1e-8
max_iters = 5000
result_color = RGBA{Float64}(colorant"#EE7733") # Tol Vibrant Orange

M = Hyperbolic(100) #SymmetricPositiveDefinite(12)
p0 = rand(M)
# signal, geodesics = artificial_signal(M, 496; a=-6.0, b=6.0, T=3)
# noise = map(p -> exp(M, p, rand(M; vector_at=p, σ=σ)), signal)
noise = [rand(M) for _ in 1:N]
# noise = rand(M)
q = rand(M)
#
# Objective, subgradient and prox
g(M, p) = 1/2length(noise) * sum(distance.(Ref(M), noise, Ref(p)).^2)
h(M, p) = α * distance(M, p, q)
f(M, p) = g(M, p) + h(M, p)
grad_g(M, p) = 1/length(noise) * sum(ManifoldDiff.grad_distance.(Ref(M), noise, Ref(p), 2))
prox_h(M, λ, p) = ManifoldDiff.prox_distance(M, α * λ, q, p, 1)
proxes = Function[(M, λ, p) -> ManifoldDiff.prox_distance(M, λ / length(noise), di, p, 2) for di in noise]
push!(proxes, (M, λ, p) -> ManifoldDiff.prox_distance(M, α * λ, q, p, 1))

@time c = cyclic_proximal_point(
    M,
    f,
    proxes,
    p0;
    stopping_criterion=StopAfterIteration(max_iters) | StopWhenChangeLess(M, atol),
    debug=[
        :Iteration,
        " | ",
        DebugProximalParameter(),
        " | ",
        (:Cost, "F(p): %1.8f "),
        " | ",
        :Change,
        "\n",
        1000,
        :Stop,
    ],
    record=[:Iteration, :Cost, :Iterate],
    return_state=true,
    # return_options=true,
)
c_result = get_solver_result(c)
c_record = get_record(c)
#
@time pg = proximal_gradient_method(
        M,
        f,
        g,
        grad_g,
        prox_h,
        p0;
        # acceleration=Manopt.ProxGradAcceleration(
        #     M; 
        #     p=p0, 
        #     # β=1.0,
        #     # β=k->(k)/(k+3),#k->(1+√(1+4k^2))/2),
        # ),
        # λ=i -> 5e-1,#4e-4,#1e-3,#0.4225,#1e-3,#1 / (i),#1/i^2,
        stepsize=
            # ConstantLength(7e-1),
            ProxGradBacktracking(; strategy=:convex, initial_stepsize=1.5),
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
                           StopAfterIteration(max_iters) | StopWhenCostNaN(),
    )
pg_result = get_solver_result(pg)
pg_record = get_record(pg)

# Calculate the minimum cost for relative error
min_cost_pg = minimum(record[2] for record in pg_record)

# Create vectors for plotting
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
plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_sqrt_pg;
    linestyle=:dash,
    linewidth=1.5,
    color=:black,
    label="O(1/√k)"
)
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

# Optional: Save the plot
# savefig(convergence_plot, joinpath(results_folder, experiment_name * "-convergence.pdf"))