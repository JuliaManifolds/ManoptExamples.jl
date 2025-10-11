# An NCRPG run on the Grassmann manifold
Hajg Jasa
2025-05-09

## Introduction

In this example we compare the Nonconvex Riemannian Proximal Gradient (NCRPG) method [BergmannJasaJohnPfeffer:2025:1](@cite) with the Cyclic Proximal Point Algorithm, which was introduced in [Bacak:2014](@cite), on the space of symmetric positive definite matrices and on hyperbolic space.
This example reproduces the results from [BergmannJasaJohnPfeffer:2025:1](@cite), Section 6.2.
The numbers may vary slightly due to having run this notebook on a different machine.

``` julia
using PrettyTables
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
```

## The Problem

Let $\mathcal M$ be a Riemannian manifold and $\{q_1,\ldots,q_N\} \in \mathcal M$ denote $N = 1000$
Gaussian random data points.
Let $g \colon \mathcal M \to \mathbb R$ be defined by

``` math
g(p) = \sum_{j = 1}^N w_j \, \mathrm{dist}(p, q_j)^2,
```

where $w_j$, $j = 1, \ldots, N$ are positive weights such that $\sum_{j = 1}^N w_j = 1$.
In our experiments, we choose the weights $w_j = \frac{1}{2N}$.
Observe that the function $g$ is strongly convex with respect to the Riemannian metric on $\mathcal M$.
The Riemannian geometric median $p^*$ of the dataset $\{q_1,\ldots,q_N\}$

``` math
\mathcal D = \{
    q_1,\ldots,q_N \, \vert \, q_j \in \mathcal M\text{ for all } j = 1,\ldots,N
\}
```

is then defined as

``` math
    p^* \in \operatorname*{arg\,min}_{p \in \mathcal M} g(p).
```

Let now $\bar q \in \mathcal M$ be a given point, and let $h \colon \mathcal M \to \mathbb R$ be defined by

``` math
h(p) = \alpha \mathrm{dist}(p, \bar q).
```

We define our total objective function as $f = g + h$.
Notice that this objective function is strongly convex with respect to the Riemannian metric on $\mathcal M$ thanks to $g$.
The goal is to find the minimizer of $f$ on $\mathcal M$, which heuristically is an interpolation between the geometric median $p^*$ and $\bar q$.

## Numerical Experiment

We initialize the experiment parameters, as well as some utility functions.

``` julia
random_seed = 100
experiment_name = "NCRPG-Grassmann"
results_folder = joinpath(@__DIR__, experiment_name)
!isdir(results_folder) && mkdir(results_folder)

atol = 1e-7
max_iters = 5000
N = 1000 # number of data points
α = 1/2 # weight for the median component (h)
δ = 1e-2 # parameter for the estimation of the constant stepsize
σ = 1.0 # standard deviation for the Gaussian data points
k_max_gr = 2.0 # maximum curvature of the Grassmann manifold
gr_dims = [(5, 2), (10, 4), (50, 10), (100, 20), (200, 40)] # dimensions of the Grassmann manifold
```

``` julia
# Objective, gradient, and proxes
g(M, p, data) = 1/2length(data) * sum(distance.(Ref(M), data, Ref(p)).^2)
grad_g(M, p, data) = 1/length(data) * sum(ManifoldDiff.grad_distance.(Ref(M), data, Ref(p), 2))
# 
h(M, p, q) = α * distance(M, p, q)
prox_h(M, λ, p, q) = ManifoldDiff.prox_distance(M, α * λ, q, p, 1)
# 
f(M, p, data, q) = g(M, p, data) + h(M, p, q)
# CPPA needs the proximal operators for the total objective
function proxes_f(data, q)
    proxes = Function[(M, λ, p) -> ManifoldDiff.prox_distance(M, λ / length(data), di, p, 2) for di in data]
    push!(proxes, (M, λ, p) -> ManifoldDiff.prox_distance(M, α * λ, q, p, 1))
    return proxes
end
# Function to shorten vectors if they are too long (for convexity reasons)
function shorten_vectors!(M, p, vectors)
    # If the i-th vector is of length greater than π/(2 * √(k_max_gr)), randomly shorten it
    # to a length between 0 and π/(2 * √(k_max_gr)) (excluded)
    for i in 1:length(vectors)
        if norm(M, p, vectors[i]) ≥ π/(2 * √(k_max_gr))
            # Randomly shorten the vector to a length between 0 and π/(2 * √(k_max_gr))
            new_length = rand() * π/(2 * √(k_max_gr))
            vectors[i] = new_length * (vectors[i] / norm(M, p, vectors[i]))
        end
    end
    return vectors
end
# Function to generate points close to the given point p
function close_point(M, p, tol; retraction_method=Manifolds.default_retraction_method(M, typeof(p)))
    X = rand(M; vector_at = p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, retraction_method)
end
# Functions for estimating the constant stepsize
ζ(δ) = π/((2 + δ) * √k_max_gr) * cot(π/((2 + δ) * √k_max_gr))
function λ_δ(δ, M, h, grad_g, p1, k_max, D, N=100)
    points = [close_point(M, p1, D/2) for _ in 1:N]
    α_g = maximum([norm(M, p, grad_g(M, p)) for p in points])
    α_1 = minimum([h(M, p) for p in points])
    α_2 = maximum([h(M, p) for p in points])
    π_k = π / √k_max
    return (√(4*(α_2 - α_1)^2 + π_k^2/(2+δ)^2 * α_g^2) - 2*(α_2 - α_1))/(2*α_g^2)
end
```

We introduce some keyword arguments for the solvers we will use in this experiment

``` julia
pgm_kwargs(initial_stepsize) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => ProximalGradientMethodBacktracking(; 
        strategy=:nonconvex, 
        initial_stepsize=initial_stepsize,
        stop_when_stepsize_less=atol,
        ),
    :stopping_criterion => StopWhenAny(
        StopWhenGradientMappingNormLess(atol), StopAfterIteration(max_iters)
    ),
]
pgm_bm_kwargs(initial_stepsize) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => ProximalGradientMethodBacktracking(; 
        strategy=:nonconvex,   
        initial_stepsize=initial_stepsize,
        stop_when_stepsize_less=atol,
        ),
    :stopping_criterion => StopWhenAny(
        StopWhenGradientMappingNormLess(atol), StopAfterIteration(max_iters)
    ), 
]
# 
pgm_kwargs_constant(stepsize) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => ConstantLength(stepsize),
    :stopping_criterion => StopWhenAny(
        StopWhenGradientMappingNormLess(atol), StopAfterIteration(max_iters)
    ),
]
pgm_bm_kwargs_constant(stepsize) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => ConstantLength(stepsize),
    :stopping_criterion => StopWhenAny(
        StopWhenGradientMappingNormLess(atol), StopAfterIteration(max_iters)
    ), 
]
# 
cppa_kwargs(M) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopWhenAny(
        StopAfterIteration(max_iters), StopWhenChangeLess(M, atol)
    ),
]
cppa_bm_kwargs(M) = [
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopWhenAny(
        StopAfterIteration(max_iters), StopWhenChangeLess(M, atol)
    ),
]
```

Before running the experiments, we initialize data collection functions that we will use later

``` julia
global col_names_1 = [
    :Dimension,
    :Time_1,
    :Iterations_1,
    :Objective_1,
    :Time_2,
    :Iterations_2,
    :Objective_2,
]
col_types_1 = [
    Int64,
    Float64,
    Int64,
    Float64,
    Float64,
    Int64,
    Float64,
]
named_tuple_1 = (; zip(col_names_1, type[] for type in col_types_1 )...)
function initialize_dataframes(results_folder, experiment_name, subexperiment_name, named_tuple_1)
    A1 = DataFrame(named_tuple_1)
    CSV.write(
        joinpath(
            results_folder,
            experiment_name * 
            "-Comparisons.csv",
        ),
        A1;
        header=false,
    )
    return A1
end
```

``` julia
function export_dataframes(M, records, times, results_folder, experiment_name, subexperiment_name, col_names_1)
    B1 = DataFrame(;
        Dimension=manifold_dimension(M),
        Time_1=times[1],
        Iterations_1=maximum(first.(records[1])),
        Objective_1=minimum([r[2] for r in records[1]]),
        Time_2=times[2],
        Iterations_2=maximum(first.(records[2])),
        Objective_2=minimum([r[2] for r in records[2]]),   
    )
    return B1
end
function write_dataframes(
    B1, 
    results_folder, 
    experiment_name, 
    subexperiment_name
)
    CSV.write(
        joinpath(
            results_folder,
            experiment_name *
            "-Comparisons.csv",
        ),
        B1;
        append=true,
    )
end
```

``` julia
subexperiment_name = "Gr"
global A1_Gr = initialize_dataframes(
    results_folder,
    experiment_name,
    subexperiment_name,
    named_tuple_1,
)

for (n, m) in gr_dims

    Random.seed!(random_seed)

    M = Grassmann(Int(n), Int(m))
    anchor = rand(M)
    vectors = [rand(M; vector_at=anchor, σ=σ) for _ in 1:N+2]
    shorten_vectors!(M, anchor, vectors)
    data = [exp(M, anchor, vectors[i]) for i in 1:N]
    q_bar = exp(M, anchor, vectors[N+1]) # point for the median component, i.e. h
    p0 = exp(M, anchor, vectors[N+2])

    g_gr(M, p) = g(M, p, data)
    h_gr(M, p) = h(M, p, q_bar)
    grad_g_gr(M, p) = grad_g(M, p, data)
    proxes_f_gr = proxes_f(data, q_bar)
    prox_h_gr(M, λ, p) = prox_h(M, λ, p, q_bar)
    f_gr(M, p) = f(M, p, data, q_bar)

    D = 2maximum([distance(M, anchor, pt) for pt in vcat(data, [q_bar], [p0])])
    L_g = 1.0 # since k_min = 0.0
    λ = minimum([λ_δ(δ, M, h_gr, grad_g_gr, anchor, k_max_gr, D, 100) for _ in 1:10]) # estimate λ_δ
    constant_stepsize = max(0, min(λ, ζ(δ)/L_g))
    initial_stepsize = 1.0

    # Optimization
    pgm_constant = proximal_gradient_method(M, f_gr, g_gr, grad_g_gr, p0; prox_nonsmooth=prox_h_gr, pgm_kwargs_constant(constant_stepsize)...)
    pgm_constant_result = get_solver_result(pgm_constant)
    pgm_constant_record = get_record(pgm_constant)

    # We can also use a backtracked stepsize
    pgm = proximal_gradient_method(M, f_gr, g_gr, grad_g_gr, p0; prox_nonsmooth=prox_h_gr, pgm_kwargs(initial_stepsize)...)
    pgm_result = get_solver_result(pgm)
    pgm_record = get_record(pgm)

    records = [
        pgm_constant_record,
        pgm_record,
    ]

    if benchmarking
        pgm_constant_bm = @benchmark proximal_gradient_method($M, $f_gr, $g_gr, $grad_g_gr, $p0; prox_nonsmooth=$prox_h_gr, $pgm_bm_kwargs_constant($constant_stepsize)...)
        pgm_bm = @benchmark proximal_gradient_method($M, $f_gr, $g_gr, $grad_g_gr, $p0; prox_nonsmooth=$prox_h_gr, $pgm_bm_kwargs($initial_stepsize)...)
        
        times = [
            median(pgm_constant_bm).time * 1e-9,
            median(pgm_bm).time * 1e-9,
        ]

        B1 = export_dataframes(
            M,
            records,
            times,
            results_folder,
            experiment_name,
            subexperiment_name,
            col_names_1,
        )

        append!(A1_Gr, B1)
        (export_table) && (write_dataframes(B1, results_folder, experiment_name, subexperiment_name))
    end
end
```

We can take a look at how the algorithms compare to each other in their performance with the following table, where columns 2 to 4 relate to the NCRPG with a constant stepsize, while columns 5 to 7 refer to a backtracked stepsize…

| **Dimension** | **Time\_1** | **Iterations\_1** | **Objective\_1** | **Time\_2** | **Iterations\_2** | **Objective\_2** |
|--------------:|------------:|------------------:|-----------------:|------------:|------------------:|-----------------:|
| 6             | 0.389674    | 90                | 0.853678         | 0.0987565   | 12                | 0.853678         |
| 24            | 0.583378    | 58                | 0.858344         | 0.180316    | 9                 | 0.858344         |
| 400           | 2.37086     | 36                | 0.868749         | 0.70433     | 6                 | 0.868749         |
| 1600          | 8.63405     | 35                | 0.871773         | 4.17984     | 6                 | 0.871773         |
| 6400          | 33.6688     | 34                | 0.873426         | 9.63182     | 5                 | 0.873426         |

## Technical details

This tutorial is cached. It was last run on the following package versions.

``` julia
using Pkg
Pkg.status()
```

    Status `~/Repositories/Julia/ManoptExamples.jl/examples/Project.toml`
      [6e4b80f9] BenchmarkTools v1.6.0
      [336ed68f] CSV v0.10.15
      [13f3f980] CairoMakie v0.15.6
      [0ca39b1e] Chairmarks v1.3.1
      [35d6a980] ColorSchemes v3.31.0
    ⌅ [5ae59095] Colors v0.12.11
      [a93c6f00] DataFrames v1.8.0
      [31c24e10] Distributions v0.25.122
      [7073ff75] IJulia v1.30.6
    ⌅ [682c06a0] JSON v0.21.4
      [8ac3fa9e] LRUCache v1.6.2
      [b964fa9f] LaTeXStrings v1.4.0
      [d3d80556] LineSearches v7.4.0
      [ee78f7c6] Makie v0.24.6
      [af67fdf4] ManifoldDiff v0.4.5
      [1cead3c2] Manifolds v0.11.0
      [3362f125] ManifoldsBase v2.0.0
      [0fc0a36d] Manopt v0.5.25
      [5b8d5e80] ManoptExamples v0.1.16 `..`
      [51fcb6bd] NamedColors v0.2.3
      [91a5bcdd] Plots v1.41.1
    ⌅ [08abe8d2] PrettyTables v2.4.0
      [6099a3de] PythonCall v0.9.28
      [f468eda6] QuadraticModels v0.9.14
      [1e40b3f8] RipQP v0.7.0
    Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated`

This tutorial was last rendered October 11, 2025, 14:11:23.

## Literature

```@bibliography
Pages = ["NCRPG-Grassmann.md"]
Canonical=false
```
