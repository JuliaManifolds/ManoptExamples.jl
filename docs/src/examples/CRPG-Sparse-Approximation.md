# A Sparse Approximation Problem on Hadamard Manifolds
Hajg Jasa, Paula John
2025-07-02

## Introduction

In this example we use the Convex Riemannian Proximal Gradient (CRPG) method [BergmannJasaJohnPfeffer:2025:2](@cite) with the Cyclic Proximal Point Algorithm, which was introduced in [Bacak:2014](@cite), on the hyperbolic space.
This example reproduces the results from [BergmannJasaJohnPfeffer:2025:2](@cite), Section 6.3.

``` julia
using PrettyTables
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
```

## The Problem

Let $\mathcal M = \mathcal H^n$ be the Hadamard manifold given by the hyperbolic space, and $\{q_1,\ldots,q_N\} \in \mathcal M$ denote $N = 1000$ Gaussian random data points.
Let $g \colon \mathcal M \to \mathbb R$ be defined by

``` math
g(p) = \frac{1}{2} \sum_{j = 1}^N w_j \, \mathrm{dist}(p, q_j)^2,
```

where $w_j$, $j = 1, \ldots, N$ are positive weights such that $\sum_{j = 1}^N w_j = 1$.
In our experiments, we choose the weights $w_j = \frac{1}{N}$.
Observe that the function $g$ is strongly convex with respect to the Riemannian metric on $\mathcal M$.

Let $h \colon \mathcal M \to \mathbb R$ be defined by

``` math
h(p) = \mu \Vert p \Vert_1
```

be the sparsity-enforcing term given by the $\ell_1$-norm, where $\mu > 0$ is a regularization parameter.

We define our total objective function as $f = g + h$.
Notice that this objective function is strongly convex with respect to the Riemannian metric on $\mathcal M$ thanks to $g$.
The goal is to find the minimizer of $f$ on $\mathcal M$, which is heuristically the point that is closest to the data points $q_j$ in the sense of the Riemannian metric on $\mathcal M$ and has a sparse representation.

## Numerical Experiment

We initialize the experiment parameters, as well as some utility functions.

``` julia
random_seed = 31
n_tests = 10 # number of tests for each parameter setting

atol = 1e-7
max_iters = 5000
N = 1000 # number of data points
dims = [2, 10, 100, 500]
μs = [0.1, 0.5, 1.0]
σ = 1.0 # standard deviation for the Gaussian random data points
```

``` julia
# Objective, gradient, and proxes
g(M, p, data) = 1/2length(data) * sum(distance.(Ref(M), data, Ref(p)).^2)
grad_g(M, p, data) = 1/length(data) * sum(ManifoldDiff.grad_distance.(Ref(M), data, Ref(p), 2))
#
# Proximal map for the $\ell_1$-norm on the hyperbolic space
function prox_l1_Hn(Hn, μ, x; t_0 = μ, max_it = 20, tol = 1e-7)
    n = manifold_dimension(Hn)
    t = t_0
    y = zeros(n+1)
    y[end] = x[end] + t
    for i in 1:n
        y[i] = sign(x[i])*max(0, abs(x[i]) - t)
    end
    y /= sqrt(abs(minkowski_metric(y, y)))
    for k in 1:max_it
        t_new = μ * sqrt(abs(minkowski_metric(x, y)^2 - 1 ))/distance(Hn, x, y)
        if abs(t_new - t) ≤ tol
            return y
        end
        y[end] = x[end] + t_new
        for i in 1:n
            y[i] = sign(x[i])*max(0, abs(x[i]) - t_new)
        end
        y /= sqrt(abs(minkowski_metric(y, y)))
        t = copy(t_new)
    end
    return y
end
h(M, p, μ) = μ * norm(p, 1)
prox_h(M, λ, p, μ) = prox_l1_Hn(M, λ * μ, p)
#
f(M, p, data, μ) = g(M, p, data) + h(M, p, μ)
# CPPA needs the proximal operators for the total objective
function proxes_f(data, μ)
    proxes = Function[(M, λ, p) -> ManifoldDiff.prox_distance(M, λ / length(data), di, p, 2) for di in data]
    push!(proxes, (M, λ, p) -> prox_l1_Hn(M, λ * μ, p))
    return proxes
end
# Function to generate points close to the given point p
function close_point(M, p, tol; retraction_method=Manifolds.default_retraction_method(M, typeof(p)))
    X = rand(M; vector_at = p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, retraction_method)
end
# Estimate Lipschitz constant of the gradient of g
function estimate_lipschitz_constant(M, g, grad_g, anchor, R, N=10_000)
    constants = []
    for i in 1:N
        p = close_point(M, anchor, R)
        q = close_point(M, anchor, R)

        push!(constants, 2/distance(M, q, p)^2 * (g(M, q) - g(M, p) - inner(M, p, grad_g(M, p), log(M, p, q))))
    end
    return maximum(constants)
end
```

We introduce some keyword arguments for the solvers we will use in this experiment

``` julia
# Keyword arguments for CRPG with a constant stepsize
pgm_kwargs_cn(constant_stepsize) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => ConstantLength(constant_stepsize),
    :stopping_criterion => StopWhenAny(
        StopWhenGradientMappingNormLess(atol), StopAfterIteration(max_iters)
    ),
]
pgm_bm_kwargs_cn(constant_stepsize) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => ConstantLength(constant_stepsize),
    :stopping_criterion => StopWhenAny(
        StopWhenGradientMappingNormLess(atol), StopAfterIteration(max_iters)
    ),
]
# Keyword arguments for CRPG with a backtracked stepsize
pgm_kwargs_bt(contraction_factor, initial_stepsize, warm_start_factor) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => ProximalGradientMethodBacktracking(;
        contraction_factor=contraction_factor,
        initial_stepsize=initial_stepsize,
        strategy=:convex,
        warm_start_factor=warm_start_factor,
    ),
    :stopping_criterion => StopWhenAny(
        StopWhenGradientMappingNormLess(atol), StopAfterIteration(max_iters)
    ),
]
pgm_bm_kwargs_bt(contraction_factor, initial_stepsize, warm_start_factor) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => ProximalGradientMethodBacktracking(;
        contraction_factor=contraction_factor,
        initial_stepsize=initial_stepsize,
        strategy=:convex,
        warm_start_factor=warm_start_factor,
    ),
    :stopping_criterion => StopWhenAny(
        StopWhenGradientMappingNormLess(atol), StopAfterIteration(max_iters)
    ),
]
# Keyword arguments for CPPA
cppa_kwargs(M) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopWhenAny(
        StopAfterIteration(max_iters), StopWhenCriterionWithIterationCondition(StopWhenChangeLess(M, 1e-5*atol), 20)
    ),
]
cppa_bm_kwargs(M) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopWhenAny(
        StopAfterIteration(max_iters), StopWhenCriterionWithIterationCondition(StopWhenChangeLess(M, 1e-5*atol), 20)
    ),
]
```

We set up some variables to collect the results of the experiments and initialize the dataframes

And run the experiments

``` julia
for n in dims
    # Set random seed for reproducibility
    Random.seed!(random_seed)

    # Define manifold
    M = Hyperbolic(n)

    for test in 1:n_tests
        # Generate random data
        anchor = rand(M)
        data = [exp(M, anchor, rand(M; vector_at=anchor, σ=σ)) for _ in 1:N]

        for (c, μ) in enumerate(μs)
            # Initialize starting point for the optimization
            p0 = rand(M)

            # Initialize functions
            g_hn(M, p) = g(M, p, data)
            grad_g_hn(M, p) = grad_g(M, p, data)
            proxes_f_hn = proxes_f(data, μ)
            prox_h_hn(M, λ, p) = prox_h(M, λ, p, μ)
            f_hn(M, p) = f(M, p, data, μ)
            #
            # Estimate stepsizes
            D = 2.05 * maximum([distance(M, p0, di) for di in vcat(data, [anchor])])
            L_g = Manopt.ζ_1(-1.0, D)
            constant_stepsize = 1/L_g
            initial_stepsize = 4 * constant_stepsize
            contraction_factor = 0.8
            warm_start_factor = 10.0
            #
            # Optimization
            # Constant stepsize
            pgm_cn = proximal_gradient_method(M, f_hn, g_hn, grad_g_hn, p0;
                prox_nonsmooth=prox_h_hn,
                pgm_kwargs_cn(constant_stepsize)...
            )
            pgm_result_cn = get_solver_result(pgm_cn)
            pgm_record_cn = get_record(pgm_cn)
            #
            # Backtracked stepsize
            pgm_bt = proximal_gradient_method(M, f_hn, g_hn, grad_g_hn, p0;
                prox_nonsmooth=prox_h_hn,
                pgm_kwargs_bt(contraction_factor, initial_stepsize, warm_start_factor)...
            )
            pgm_result_bt = get_solver_result(pgm_bt)
            pgm_record_bt = get_record(pgm_bt)
            #
            # CPPA
            cppa = cyclic_proximal_point(M, f_hn, proxes_f_hn, p0; cppa_kwargs(M)...)
            cppa_result = get_solver_result(cppa)
            cppa_record = get_record(cppa)
            #
            # Benchmark the algorithms
            # Constant stepsize
            pgm_bm_cn = @benchmark proximal_gradient_method($M, $f_hn, $g_hn, $grad_g_hn, $p0;
                prox_nonsmooth=$prox_h_hn,
                $pgm_bm_kwargs_cn($constant_stepsize)...
            )
            # Backtracked stepsize
            pgm_bm_bt = @benchmark proximal_gradient_method($M, $f_hn, $g_hn, $grad_g_hn, $p0;
                prox_nonsmooth=$prox_h_hn,
                $pgm_bm_kwargs_bt($contraction_factor, $initial_stepsize, $warm_start_factor)...
            )
            # CPPA
            cppa_bm = @benchmark cyclic_proximal_point($M, $f_hn, $proxes_f_hn, $p0; cppa_bm_kwargs($M)...)
            #
            # Collect times
            time_pgm_cn = time(median(pgm_bm_cn)) * 1e-9
            time_pgm_bt = time(median(pgm_bm_bt)) * 1e-9
            time_cppa = time(median(cppa_bm)) * 1e-9
            time_pgm_cn_means[c] += time_pgm_cn
            time_pgm_bt_means[c] += time_pgm_bt
            time_cppa_means[c] += time_cppa
            #
            # Collect sparsities
            sparsity_pgm_cn = sum(abs.(pgm_result_cn) .< atol)/n
            sparsity_pgm_bt = sum(abs.(pgm_result_bt) .< atol)/n
            sparsity_cppa = sum(abs.(cppa_result) .< atol)/n
            sparsity_pgm_cn_means[c] += sparsity_pgm_cn
            sparsity_pgm_bt_means[c] += sparsity_pgm_bt
            sparsity_cppa_means[c] += sparsity_cppa
            #
            # Collect objective values
            objective_pgm_cn = f_hn(M, pgm_result_cn)
            objective_pgm_bt = f_hn(M, pgm_result_bt)
            objective_cppa = f_hn(M, cppa_result)
            objective_pgm_cn_means[c] += objective_pgm_cn
            objective_pgm_bt_means[c] += objective_pgm_bt
            objective_cppa_means[c] += objective_cppa
            #
            # Collect iterations
            iterations_pgm_cn = length(pgm_record_cn)
            iterations_pgm_bt = length(pgm_record_bt)
            iterations_cppa = length(cppa_record)
            iterations_pgm_cn_means[c] += iterations_pgm_cn
            iterations_pgm_bt_means[c] += iterations_pgm_bt
            iterations_cppa_means[c] += iterations_cppa
        end
    end
    for (c, μ) in enumerate(μs)
        push!(df_pgm_cn,
            [
                μ, n, iterations_pgm_cn_means[c]/n_tests, time_pgm_cn_means[c]/n_tests, objective_pgm_cn_means[c]/n_tests, sparsity_pgm_cn_means[c]/n_tests
            ]
        )
        push!(df_pgm_bt,
            [
                μ, n, iterations_pgm_bt_means[c]/n_tests, time_pgm_bt_means[c]/n_tests, objective_pgm_bt_means[c]/n_tests, sparsity_pgm_bt_means[c]/n_tests
            ]
        )
        push!(df_cppa,
            [
                μ, n, iterations_cppa_means[c]/n_tests, time_cppa_means[c]/n_tests, objective_cppa_means[c]/n_tests, sparsity_cppa_means[c]/n_tests
            ]
        )
    end
    #
    # Reset data collection variables
    iterations_pgm_cn_means .= zeros(length(μs))
    iterations_pgm_bt_means .= zeros(length(μs))
    iterations_cppa_means .= zeros(length(μs))
    time_pgm_cn_means .= zeros(length(μs))
    time_pgm_bt_means .= zeros(length(μs))
    time_cppa_means .= zeros(length(μs))
    sparsity_pgm_cn_means .= zeros(length(μs))
    sparsity_pgm_bt_means .= zeros(length(μs))
    sparsity_cppa_means .= zeros(length(μs))
    objective_pgm_cn_means .= zeros(length(μs))
    objective_pgm_bt_means .= zeros(length(μs))
    objective_cppa_means .= zeros(length(μs))
end
```

We export the results to CSV files

``` julia
# Sort the dataframes by the parameter μ and create the final results dataframes
df_pgm_cn = sort(df_pgm_cn, :μ)
df_pgm_bt = sort(df_pgm_bt, :μ)
df_cppa = sort(df_cppa, :μ)
df_results_time_iter = DataFrame(
    μ             = df_pgm_cn.μ,
    n             = Int.(df_pgm_cn.n),
    CRPG_iter     = Int.(round.(df_pgm_cn.iterations, digits = 0)),
    CRPG_time     = df_pgm_cn.time,
    CRPG_bt_iter  = Int.(round.(df_pgm_bt.iterations, digits = 0)),
    CRPG_bt_time  = df_pgm_bt.time,
    CPPA_iter  = Int.(round.(df_cppa.iterations, digits = 0)),
    CPPA_time     = df_cppa.time,
)
df_results_obj_spar = DataFrame(
    μ               = df_pgm_cn.μ,
    n               = Int.(df_pgm_cn.n),
    CRPG_obj       = df_pgm_cn.objective,
    CRPG_sparsity  = df_pgm_cn.sparsity,
    CRPG_bt_obj    = df_pgm_bt.objective,
    CRPG_bt_sparsity = df_pgm_bt.sparsity,
    CPPA_obj         = df_cppa.objective,
    CPPA_sparsity    = df_cppa.sparsity,
)
# Write the results to CSV files
CSV.write(joinpath(results_folder, "results-Hn-time-iter-$(n_tests)-$(dims[end]).csv"), df_results_time_iter)
CSV.write(joinpath(results_folder, "results-Hn-obj-spar-$(n_tests)-$(dims[end]).csv"), df_results_obj_spar)
```

We can take a look at how the algorithms compare to each other in their performance with the following tables.
First, we look at the time and number of iterations for each algorithm.

| **μ** | **n** | **CRPG_const_iter** | **CRPG_const_time** | **CRPG_bt_iter** | **CRPG_bt_time** | **CPPA_iter** | **CPPA_time** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 2 | 151 | 0.0310879 | 36 | 0.0109412 | 5000 | 2.79486 |
| 0.1 | 10 | 71 | 0.0188336 | 16 | 0.00592263 | 5000 | 3.10504 |
| 0.1 | 100 | 36 | 0.0265803 | 883 | 3.95681 | 5000 | 5.75863 |
| 0.1 | 500 | 31 | 0.120596 | 4614 | 177.527 | 5000 | 26.0805 |
| 0.5 | 2 | 114 | 0.0226986 | 28 | 0.00840331 | 5000 | 2.79979 |
| 0.5 | 10 | 59 | 0.0133695 | 11 | 0.00414099 | 4502 | 2.78999 |
| 0.5 | 100 | 33 | 0.0230144 | 616 | 2.64535 | 5000 | 5.72404 |
| 0.5 | 500 | 18 | 0.0729229 | 2588 | 125.64 | 1515 | 7.97267 |
| 1.0 | 2 | 47 | 0.00961312 | 11 | 0.0035858 | 2013 | 1.13444 |
| 1.0 | 10 | 32 | 0.00719324 | 8 | 0.00325584 | 2511 | 1.54804 |
| 1.0 | 100 | 14 | 0.00937349 | 297 | 1.30338 | 1018 | 1.15093 |
| 1.0 | 500 | 8 | 0.0304811 | 576 | 29.301 | 22 | 0.112924 |

Second, we look at the objective values and sparsity of the solutions found by each algorithm.

| **μ** | **n** | **CRPG_const_obj** | **CRPG_const_spar** | **CRPG_bt_obj** | **CRPG_bt_spar** | **CPPA_obj** | **CPPA_spar** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 2 | 2.57211 | 0.05 | 2.57211 | 0.05 | 2.57211 | 0.05 |
| 0.1 | 10 | 5.97048 | 0.08 | 5.97048 | 0.08 | 5.97048 | 0.08 |
| 0.1 | 100 | 50.5157 | 0.257 | 50.5157 | 0.257 | 50.5157 | 0.257 |
| 0.1 | 500 | 250.668 | 0.4988 | 250.668 | 0.499 | 250.668 | 0.5 |
| 0.5 | 2 | 3.27837 | 0.15 | 3.27837 | 0.15 | 3.27837 | 0.15 |
| 0.5 | 10 | 6.75073 | 0.59 | 6.75073 | 0.59 | 6.75074 | 0.6 |
| 0.5 | 100 | 51.3055 | 0.834 | 51.3055 | 0.834 | 51.3055 | 0.834 |
| 0.5 | 500 | 251.35 | 0.9544 | 251.35 | 0.9544 | 251.351 | 0.9644 |
| 1.0 | 2 | 3.86482 | 0.8 | 3.86482 | 0.8 | 3.86482 | 0.8 |
| 1.0 | 10 | 7.36369 | 0.89 | 7.36369 | 0.89 | 7.36369 | 0.89 |
| 1.0 | 100 | 51.8631 | 0.986 | 51.8631 | 0.986 | 51.8632 | 0.992 |
| 1.0 | 500 | 251.866 | 0.9992 | 251.866 | 0.9992 | 251.866 | 1.0 |

## Technical details

This tutorial is cached. It was last run on the following package versions.

    Status `~/Repositories/Julia/ManoptExamples.jl/examples/Project.toml`
      [6e4b80f9] BenchmarkTools v1.8.0
      [336ed68f] CSV v0.10.16
      [13f3f980] CairoMakie v0.15.13
      [0ca39b1e] Chairmarks v1.3.1
      [35d6a980] ColorSchemes v3.31.0
      [5ae59095] Colors v0.13.1
      [a93c6f00] DataFrames v1.8.2
      [31c24e10] Distributions v0.25.129
      [e9467ef8] GLMakie v0.13.13
      [5c1252a2] GeometryBasics v0.5.11
      [4d00f742] GeometryTypes v0.8.5
      [7073ff75] IJulia v1.34.4
      [682c06a0] JSON v1.6.1
      [8ac3fa9e] LRUCache v1.6.2
      [b964fa9f] LaTeXStrings v1.4.0
      [d3d80556] LineSearches v7.7.1
      [ee78f7c6] Makie v0.24.13
      [7351309b] ManifoldAsymptote v0.1.0
      [af67fdf4] ManifoldDiff v0.4.5
      [9d80ff41] ManifoldMakie v0.1.2
      [1cead3c2] Manifolds v0.11.28
      [3362f125] ManifoldsBase v2.5.0
    ⌃ [0fc0a36d] Manopt v0.6.2
      [5b8d5e80] ManoptExamples v0.1.20 `..`
      [51fcb6bd] NamedColors v0.2.3
      [6fe1bfb0] OffsetArrays v1.17.0
      [91a5bcdd] Plots v1.41.6
      [08abe8d2] PrettyTables v3.4.2
      [6099a3de] PythonCall v0.9.35
      [f468eda6] QuadraticModels v0.9.16
      [731186ca] RecursiveArrayTools v4.3.4
      [1e40b3f8] RipQP v0.7.0
    Info Packages marked with ⌃ have new versions available and may be upgradable.

This tutorial was last rendered July 22, 2026, 1:31:22.

## Literature

```@bibliography
Pages = ["CRPG-Sparse-Approximation.md"]
Canonical=false
```
