#
# Solving the spectral procrustes problem
# comparing the convex bundle method to
# 1. the proximal bundle method
# 2. the subgradient method
# 3. CPPA
#
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots
using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
#
#
# Settings
experiment_name = "Spectral_Procrustes"
results_folder = joinpath(@__DIR__, experiment_name)
figures_folder = joinpath(@__DIR__, experiment_name, "figures")
export_table = false
benchmark = true
show_plot = false
!isdir(results_folder) && mkdir(results_folder)
!isdir(figures_folder) && mkdir(figures_folder)
#
# Experiment parameters
Random.seed!(33)
n = 1000
d = 250
A = rand(n, d)
B = randn(n, d)
tol = 1e-8
function orthogonal_procrustes(A, B)
    s = svd((A' * B)')
    R = s.U * s.Vt
    return R
end
#
# Algorithm parameters
bundle_cap = 25
max_iters = 5000
δ = 0.0 #1e-2 # Update parameter for μ
μ = 50.0 # Initial proximal parameter for the proximal bundle method
k_max = 1 / 4
k_min = 0.0
diam = π / (20 * √k_max)
#
# Manifolds and data
M = SpecialOrthogonal(d)
p0 = orthogonal_procrustes(A, B)
project!(M, p0, p0)
#
# Objective and Euclidean subgradient
f(M, p) = opnorm(A - B * p)
function ∂ₑf(M, p)
    cost_svd = svd(A - B * p)
    # # find all maxima in S – since S is sorted, these are the first n ones - improve?
    indices = [i for (i, v) in enumerate(cost_svd.S) if abs(v - cost_svd.S[1]) < eps()]
    # We could extend this, since any linear combination (coefficients less than one in sum) is also in the subgradient
    ind = rand(indices)
    return -B' * (cost_svd.U[:, ind] * cost_svd.Vt[ind, :]')
end
rpb = Manifolds.RiemannianProjectionBackend(
    Manifolds.ExplicitEmbeddedBackend(M; gradient=∂ₑf)
)
∂f(M, p) = Manifolds.gradient(M, f, p, rpb)
dom(M, p) = distance(M, p, p0) < diam / 2 ? true : false
#riemannian_gradient(M, p, ∂ₑf(get_embedding(M), embed(M, p)))
#
# Optimization
println("\nConvex Bundle Method")
@time b = convex_bundle_method(
    M,
    f,
    ∂f,
    p0;
    bundle_cap=bundle_cap,
    k_max=k_max,
    k_min=k_min,
    domain=dom,
    diameter=diam,
    count=[:Cost, :SubGradient],
    cache=(:LRU, [:Cost, :SubGradient], 50),
    stopping_criterion=StopWhenLagrangeMultiplierLess(tol) | StopAfterIteration(max_iters),
    debug=[
        :Iteration,
        (:Cost, "F(p): %1.16f "),
        (:ξ, "ξ: %1.8f "),
        (:ε, "ε: %1.8f "),
        (:ϱ, "ϱ: %1.4f "),
        (:last_stepsize, "step size: %1.8f "),
        (:null_stepsize, "null step size: %1.8f"),
        :WarnBundle,
        :Stop,
        10,
        "\n",
    ],
    record=[:Iteration, :Cost, :Iterate],
    return_state=true,
)
b_result = get_solver_result(b)
b_record = get_record(b)
#
println("\nProx Bundle Method")
@time p = proximal_bundle_method(
    M,
    f,
    ∂f,
    p0;
    # ε=ε,
    # δ=δ,
    # μ=μ,
    bundle_size=bundle_cap,
    count=[:Cost, :SubGradient],
    cache=(:LRU, [:Cost, :SubGradient], 50),
    stopping_criterion=StopWhenLagrangeMultiplierLess(tol) | StopAfterIteration(max_iters),
    debug=[
        :Iteration,
        :Stop,
        (:Cost, "F(p): %1.16f "),
        (:ν, "ν: %1.16f "),
        (:c, "c: %1.16f "),
        (:μ, "μ: %1.8f "),
        # (:α, "α: %1.8f "),
        :Stop,
        :WarnBundle,
        10,
        "\n",
    ],
    record=[:Iteration, :Cost, :Iterate],
    return_state=true,
)
p_result = get_solver_result(p)
p_record = get_record(p)
println("\nSubgradient Method")
@time s = subgradient_method(
    M,
    f,
    ∂f,
    p0;
    count=[:Cost, :SubGradient],
    cache=(:LRU, [:Cost, :SubGradient], 50),
    stepsize=DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
    stopping_criterion=StopWhenSubgradientNormLess(√tol) | StopAfterIteration(max_iters),
    debug=[:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
    record=[:Iteration, :Cost, :p_star],
    return_state=true,
)
s_result = get_solver_result(s)
s_record = get_record(s)
experiments = ["RCBM", "PBA", "SGM"]
records = [b_record, p_record, s_record]
results = [b_result, p_result, s_result]
#
# Benchmarking
if benchmark
    p_bm = @benchmark proximal_bundle_method(
        $M,
        $f,
        $∂f,
        $p0;
        # ε=$ε,
        # δ=$δ,
        # μ=$μ,
        cache=(:LRU, [:Cost, :SubGradient], 50),
        stopping_criterion=StopWhenLagrangeMultiplierLess($tol) |
                           StopAfterIteration($max_iters),
    )
    b_bm = @benchmark convex_bundle_method(
        $M,
        $f,
        $∂f,
        $p0;
        k_max=$k_max,
        k_min=$k_min,
        domain=$dom,
        diameter=$diam,
        cache=(:LRU, [:Cost, :SubGradient], 50),
        stopping_criterion=StopWhenLagrangeMultiplierLess($tol) |
                           StopAfterIteration($max_iters),
    )
    s_bm = @benchmark subgradient_method(
        $M,
        $f,
        $∂f,
        $p0;
        # count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
        stepsize=DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
        stopping_criterion=StopWhenSubgradientNormLess(√$tol) |
                           StopAfterIteration($max_iters),
    )
    #
    times = [median(b_bm).time * 1e-9, median(p_bm).time * 1e-9, median(s_bm).time * 1e-9]
    #
    # Finalize - export costs
    if export_table
        D = cat(
            experiments,
            [maximum(first.(record)) for record in records],
            [t for t in times],
            [minimum([f(M, r[3]) for r in record]) for record in records];
            dims=2,
        )
        CSV.write(
            joinpath(results_folder, experiment_name * "-comparisons.csv"),
            DataFrame(D, :auto);
            header=["Algorithm", "Iterations", "Time (s)", "Objective"],
        )
    end
end
# Cost plot and export
if show_plot
    fig = plot(; xscale=:log10)
end
for (record, result, experiment) in zip(records, results, experiments)
    C1 = [0.5 f(M, p0)]
    C = cat(first.(record), [f(M, r[3]) for r in record]; dims=2)
    data = vcat(C1, C)
    if export_table
        CSV.write(
            joinpath(results_folder, experiment_name * "_" * experiment * "-result.csv"),
            DataFrame(data, :auto);
            header=["i", "cost"],
        )
    end
    if show_plot
        plot!(fig, data[:, 1], data[:, 2]; label=experiment)
    end
end
show_plot && fig
