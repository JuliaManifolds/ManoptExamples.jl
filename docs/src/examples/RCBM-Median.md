# A comparison of the RCBM with the PBA and the SGM for the Riemannian median
Hajg Jasa
2024-06-27

## Introduction

In this example we compare the Riemannian Convex Bundle Method (RCBM) [BergmannHerzogJasa:2024](@cite)
with the Proximal Bundle Algorithm, which was introduced in [HoseiniMonjeziNobakhtianPouryayevali:2021](@cite), and with the Subgradient Method (SGM), introduced in \[FerreiraOliveira:1998:1\], to find the Riemannian median.
This example reproduces the results from [BergmannHerzogJasa:2024](@cite), Section 5.
The runtimes reported in the tables are measured in seconds.

``` julia
using PrettyTables
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots
using QuadraticModels, RipQP
using LinearAlgebra, LRUCache, Random
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
```

Let $\mathcal M$ be a Hadamard manifold and $\{q_1,\ldots,q_N\} \in \mathcal M$ denote $N = 1000$
Gaussian random data points.
Let $f \colon \mathcal M \to \mathbb R$ be defined by

``` math
f(p) = \sum_{j = 1}^N w_j \, \mathrm{dist}(p, q_j),
```

where $w_j$, $j = 1, \ldots, N$ are positive weights such that $\sum_{j = 1}^N w_j = 1$.

The Riemannian geometric median $p^*$ of the dataset

``` math
\mathcal D = \{
    q_1,\ldots,q_N \, \vert \, q_j \in \mathcal M\text{ for all } j = 1,\ldots,N
\}
```

is then defined as

``` math
    p^* \coloneqq \operatorname*{arg\,min}_{p \in \mathcal M} f(p),
```

where equality is justified since $p^*$ is uniquely determined on Hadamard manifolds. In our experiments, we choose the weights $w_j = \frac{1}{N}$.

We initialize the experiment parameters, as well as utility functions.

``` julia
experiment_name = "RCBM-Median"
results_folder = joinpath(@__DIR__, experiment_name)
!isdir(results_folder) && mkdir(results_folder)
seed_argument = 57

atol = 1e-8
N = 1000 # number of data points
spd_dims = [2, 5, 10, 15]
hn_sn_dims = [1, 2, 5, 10, 15]

# Generate a point that is at most `tol` close to the point `p` on `M`
function close_point(M, p, tol; retraction_method=Manifolds.default_retraction_method(M, typeof(p)))
    X = rand(M; vector_at = p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, retraction_method)
end
```

``` julia
# Objective and subdifferential
f(M, p, data) = sum(1 / length(data) * distance.(Ref(M), Ref(p), data))
domf(M, p, centroid, diameter) = distance(M, p, centroid) < diameter / 2 ? true : false
function ∂f(M, p, data, atol=atol)
    return sum(
        1 / length(data) *
        ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=atol),
    )
end
```

``` julia
maxiter = 5000
rcbm_kwargs(diameter, domf, k_max, k_min) = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :count => [:Cost, :SubGradient],
    :domain => domf,
    :debug => [
        :Iteration,
        (:Cost, "F(p): %1.16f "),
        (:ξ, "ξ: %1.8f "),
        (:last_stepsize, "step size: %1.8f"),
        :Stop,
        1000,
        "\n",
    ],
    :diameter => diameter,
    :k_max => k_max,
    :k_min => k_min,
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
rcbm_bm_kwargs(diameter, domf, k_max, k_min) = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :diameter => diameter,
    :domain => domf,
    :k_max => k_max,
    :k_min => k_min,
]
pba_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :count => [:Cost, :SubGradient],
    :debug => [
        :Iteration,
        :Stop,
        (:Cost, "F(p): %1.16f "),
        (:ν, "ν: %1.16f "),
        (:c, "c: %1.16f "),
        (:μ, "μ: %1.8f "),
        :Stop,
        1000,
        "\n",
    ],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(maxiter),
]
pba_bm_kwargs = [:cache => (:LRU, [:Cost, :SubGradient], 50),]
sgm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :count => [:Cost, :SubGradient],
    :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
    :stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(maxiter),
]
sgm_bm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stepsize => DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(maxiter),
]
```

Before running the experiments, we initialize data collection functions that we will use later

``` julia
global col_names_1 = [
    :Dimension,
    :Iterations_1,
    :Time_1,
    :Objective_1,
    :Iterations_2,
    :Time_2,
    :Objective_2,
]
col_types_1 = [
    Int64,
    Int64,
    Float64,
    Float64,
    Int64,
    Float64,
    Float64,
]
named_tuple_1 = (; zip(col_names_1, type[] for type in col_types_1 )...)
global col_names_2 = [
    :Dimension,
    :Iterations,
    :Time,
    :Objective,
]
col_types_2 = [
    Int64,
    Int64,
    Float64,
    Float64,
]
named_tuple_2 = (; zip(col_names_2, type[] for type in col_types_2 )...)
function initialize_dataframes(results_folder, experiment_name, subexperiment_name, named_tuple_1, named_tuple_2)
    A1 = DataFrame(named_tuple_1)
    CSV.write(
        joinpath(
            results_folder,
            experiment_name * "_$subexperiment_name" * "-Comparisons-Convex-Prox.csv",
        ),
        A1;
        header=false,
    )
    A2 = DataFrame(named_tuple_2)
    CSV.write(
        joinpath(
            results_folder,
            experiment_name * "_$subexperiment_name" * "-Comparisons-Subgrad.csv",
        ),
        A2;
        header=false,
    )
    return A1, A2
end
```

``` julia
function export_dataframes(M, records, times, results_folder, experiment_name, subexperiment_name, col_names_1, col_names_2)
    B1 = DataFrame(;
        Dimension=manifold_dimension(M),
        Iterations_1=maximum(first.(records[1])),
        Time_1=times[1],
        Objective_1=minimum([r[2] for r in records[1]]),
        Iterations_2=maximum(first.(records[2])),
        Time_2=times[2],
        Objective_2=minimum([r[2] for r in records[2]]),
    )
    B2 = DataFrame(;
        Dimension=manifold_dimension(M),
        Iterations=maximum(first.(records[3])),
        Time=times[3],
        Objective=minimum([r[2] for r in records[3]]),
    )
    return B1, B2
end
function write_dataframes(B1, B2, results_folder, experiment_name, subexperiment_name)
    CSV.write(
        joinpath(
            results_folder,
            experiment_name *
            "_$subexperiment_name" *
            "-Comparisons-Convex-Prox.csv",
        ),
        B1;
        append=true,
    )
    CSV.write(
        joinpath(
            results_folder,
            experiment_name *
            "_$subexperiment_name" *
            "-Comparisons-Subgrad.csv",
        ),
        B2;
        append=true,
    )
end
```

## The Median on the Hyperboloid Model

``` julia
subexperiment_name = "Hn"
k_max_hn = -1.0
k_min_hn = -1.0

global A1, A2 = initialize_dataframes(
    results_folder,
    experiment_name,
    subexperiment_name,
    named_tuple_1,
    named_tuple_2
)

for n in hn_sn_dims
    Random.seed!(seed_argument)

    M = Hyperbolic(Int(2^n))
    data_hn = [rand(M) for _ in 1:N]
    dists = [distance(M, z, y) for z in data_hn, y in data_hn]
    diameter_hn = 2 * maximum(dists)
    p0 = data_hn[minimum(Tuple(findmax(dists)[2]))]

    f_hn(M, p) = f(M, p, data_hn)
    domf_hn(M, p) = domf(M, p, p0, diameter_hn)
    ∂f_hn(M, p) = ∂f(M, p, data_hn, atol)

    # Optimization
    rcbm = convex_bundle_method(M, f_hn, ∂f_hn, p0; rcbm_kwargs(diameter_hn, domf_hn, k_max_hn, k_min_hn)...)
    rcbm_result = get_solver_result(rcbm)
    rcbm_record = get_record(rcbm)

    pba = proximal_bundle_method(M, f_hn, ∂f_hn, p0; pba_kwargs...)
    pba_result = get_solver_result(pba)
    pba_record = get_record(pba)

    sgm = subgradient_method(M, f_hn, ∂f_hn, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        rcbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        rcbm_bm = @benchmark convex_bundle_method($M, $f_hn, $∂f_hn, $p0; rcbm_bm_kwargs($diameter_hn, $domf_hn, $k_max_hn, $k_min_hn)...)
        pba_bm = @benchmark proximal_bundle_method($M, $f_hn, $∂f_hn, $p0; $pba_bm_kwargs...)
        sgm_bm = @benchmark subgradient_method($M, $f_hn, $∂f_hn, $p0; $sgm_bm_kwargs...)
        
        times = [
            median(rcbm_bm).time * 1e-9,
            median(pba_bm).time * 1e-9,
            median(sgm_bm).time * 1e-9,
        ]

        B1, B2 = export_dataframes(
            M,
            records,
            times,
            results_folder,
            experiment_name,
            subexperiment_name,
            col_names_1,
            col_names_2,
        )

        append!(A1, B1)
        append!(A2, B2)
        (export_table) && (write_dataframes(B1, B2, results_folder, experiment_name, subexperiment_name))
    end
end
```

We can take a look at how the algorithms compare to each other in their performance with the following table, where columns 2 to 4 relate to the RCBM, while columns 5 to 7 refer to the PBA…

| Dimension | Iterations_1 |     Time_1 | Objective_1 | Iterations_2 |   Time_2 | Objective_2 |
|-----------|--------------|------------|-------------|--------------|----------|-------------|
|         2 |            9 | 0.00523775 |     1.05192 |          251 | 0.132011 |     1.05192 |
|         4 |            8 | 0.00469981 |     1.07516 |          230 | 0.132091 |     1.07516 |
|        32 |           15 |  0.0151958 |     1.08559 |          234 | 0.180374 |     1.08559 |
|      1024 |           16 |   0.284984 |     1.09706 |          234 |  4.00771 |     1.09706 |
|     32768 |           16 |    7.34017 |      1.0681 |          229 |  91.2803 |      1.0681 |

… Whereas the following table refers to the SGM

| Dimension | Iterations |       Time | Objective |
|-----------|------------|------------|-----------|
|         2 |         18 | 0.00811254 |   1.04748 |
|         4 |         19 | 0.00953129 |   1.05518 |
|        32 |         25 |  0.0208788 |   1.08559 |
|      1024 |         23 |   0.400038 |   1.09706 |
|     32768 |         21 |    8.81869 |   1.06488 |

## The Median on the Symmetric Positive Definite Matrix Space

``` julia
subexperiment_name = "SPD"
k_max_spd = 0.0
k_min_spd = -1/2

global A1_SPD, A2_SPD = initialize_dataframes(
    results_folder,
    experiment_name,
    subexperiment_name,
    named_tuple_1,
    named_tuple_2
)

for n in spd_dims
    Random.seed!(seed_argument)

    M = SymmetricPositiveDefinite(Int(n))
    data_spd = [rand(M) for _ in 1:N]
    dists = [distance(M, z, y) for z in data_spd, y in data_spd]
    diameter_spd = 2 * maximum(dists)
    p0 = data_spd[minimum(Tuple(findmax(dists)[2]))]
    
    f_spd(M, p) = f(M, p, data_spd)
    domf_spd(M, p) = domf(M, p, p0, diameter_spd)
    ∂f_spd(M, p) = ∂f(M, p, data_spd, atol)

    # Optimization
    rcbm = convex_bundle_method(M, f_spd, ∂f_spd, p0; rcbm_kwargs(diameter_spd, domf_spd, k_max_spd, k_min_spd)...)
    rcbm_result = get_solver_result(rcbm)
    rcbm_record = get_record(rcbm)

    pba = proximal_bundle_method(M, f_spd, ∂f_spd, p0; pba_kwargs...)
    pba_result = get_solver_result(pba)
    pba_record = get_record(pba)

    sgm = subgradient_method(M, f_spd, ∂f_spd, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        rcbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        rcbm_bm = @benchmark convex_bundle_method($M, $f_spd, $∂f_spd, $p0; rcbm_bm_kwargs($diameter_spd, $domf_spd, $k_max_spd, $k_min_spd)...)
        pba_bm = @benchmark proximal_bundle_method($M, $f_spd, $∂f_spd, $p0; $pba_bm_kwargs...)
        sgm_bm = @benchmark subgradient_method($M, $f_spd, $∂f_spd, $p0; $sgm_bm_kwargs...)

        times = [
            median(rcbm_bm).time * 1e-9,
            median(pba_bm).time * 1e-9,
            median(sgm_bm).time * 1e-9,
        ]

        B1_SPD, B2_SPD = export_dataframes(
            M,
            records,
            times,
            results_folder,
            experiment_name,
            subexperiment_name,
            col_names_1,
            col_names_2,
        )

        append!(A1_SPD, B1_SPD)
        append!(A2_SPD, B2_SPD)
        (export_table) && (write_dataframes(B1_SPD, B2_SPD, results_folder, experiment_name, subexperiment_name))
    end
end
```

We can take a look at how the algorithms compare to each other in their performance with the following table, where columns 2 to 4 relate to the RCBM, while columns 5 to 7 refer to the PBA…

| Dimension | Iterations_1 |   Time_1 | Objective_1 | Iterations_2 |   Time_2 | Objective_2 |
|-----------|--------------|----------|-------------|--------------|----------|-------------|
|         3 |           43 | 0.303751 |    0.260846 |           57 | 0.441796 |    0.260846 |
|        15 |           49 |  2.01407 |    0.436536 |           75 |  1.74885 |    0.436536 |
|        55 |           15 |  1.30749 |    0.618059 |           89 |  6.15426 |    0.618059 |
|       120 |            6 |  1.20377 |    0.764031 |          123 |  15.4064 |    0.764031 |

… Whereas the following table refers to the SGM

| Dimension | Iterations |    Time | Objective |
|-----------|------------|---------|-----------|
|         3 |       4629 | 46.5469 |  0.260846 |
|        15 |       1727 | 40.4873 |  0.436536 |
|        55 |        776 | 53.3628 |  0.618059 |
|       120 |        438 | 53.5932 |  0.764031 |

## The Median on the Sphere

For the last experiment, note that a major difference here is that the sphere has constant positive sectional curvature equal to $1$. In this case, we lose the global convexity of the Riemannian distance and thus of the objective. Minimizers still exist, but they may, in general, be non-unique.

``` julia
subexperiment_name = "Sn"
k_max_sn = 1.0
k_min_sn = 1.0
diameter_sn = π / 3

global A1_Sn, A2_Sn = initialize_dataframes(
    results_folder,
    experiment_name,
    subexperiment_name,
    named_tuple_1,
    named_tuple_2
)

for n in hn_sn_dims
    Random.seed!(seed_argument)

    M = Sphere(Int(2^n))
    north = [0.0 for _ in 1:manifold_dimension(M)]
    push!(north, 1.0)
    data_sn = [close_point(M, north, diameter_sn / 2)]
    distance(M, data_sn[1], north) < diameter_sn / 2 ? pop!(data_sn) : nothing
    while length(data_sn) < N
        q = close_point(M, north, diameter_sn / 2)
        distance(M, q, north) < diameter_sn / 2 ? push!(data_sn, q) : nothing 
    end
    dists = [distance(M, z, y) for z in data_sn, y in data_sn]
    p0 = data_sn[minimum(Tuple(findmax(dists)[2]))]

    f_sn(M, p) = f(M, p, data_sn)
    domf_sn(M, p) = domf(M, p, north, diameter_sn)
    ∂f_sn(M, p) = ∂f(M, p, data_sn, atol)

    # Optimization
    rcbm = convex_bundle_method(M, f_sn, ∂f_sn, p0; rcbm_kwargs(diameter_sn, domf_sn, k_max_sn, k_min_sn)...)
    rcbm_result = get_solver_result(rcbm)
    rcbm_record = get_record(rcbm)

    pba = proximal_bundle_method(M, f_sn, ∂f_sn, p0; pba_kwargs...)
    pba_result = get_solver_result(pba)
    pba_record = get_record(pba)

    sgm = subgradient_method(M, f_sn, ∂f_sn, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        rcbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        rcbm_bm = @benchmark convex_bundle_method($M, $f_sn, $∂f_sn, $p0; rcbm_bm_kwargs($diameter_sn, $domf_sn, $k_max_sn, $k_min_sn)...)
        pba_bm = @benchmark proximal_bundle_method($M, $f_sn, $∂f_sn, $p0; $pba_bm_kwargs...)
        sgm_bm = @benchmark subgradient_method($M, $f_sn, $∂f_sn, $p0; $sgm_bm_kwargs...)

        times = [
            median(rcbm_bm).time * 1e-9,
            median(pba_bm).time * 1e-9,
            median(sgm_bm).time * 1e-9,
        ]

        B1_Sn, B2_Sn = export_dataframes(
            M,
            records,
            times,
            results_folder,
            experiment_name,
            subexperiment_name,
            col_names_1,
            col_names_2,
        )

        append!(A1_Sn, B1_Sn)
        append!(A2_Sn, B2_Sn)
        (export_table) && (write_dataframes(B1_Sn, B2_Sn, results_folder, experiment_name, subexperiment_name))
    end
end
```

We can take a look at how the algorithms compare to each other in their performance with the following table, where columns 2 to 4 relate to the RCBM, while columns 5 to 7 refer to the PBA…

| Dimension | Iterations_1 |    Time_1 | Objective_1 | Iterations_2 |    Time_2 | Objective_2 |
|-----------|--------------|-----------|-------------|--------------|-----------|-------------|
|         2 |           43 | 0.0158139 |    0.258898 |           71 | 0.0184203 |    0.258898 |
|         4 |           74 | 0.0230197 |    0.253525 |           62 | 0.0168082 |    0.253525 |
|        32 |          102 |  0.043011 |    0.259886 |           64 | 0.0272739 |    0.259886 |
|      1024 |          103 |  0.890434 |    0.266993 |           68 |  0.622527 |    0.266993 |
|     32768 |           80 |   27.1998 |    0.259302 |           65 |   29.1502 |    0.259302 |

… Whereas the following table refers to the SGM

| Dimension | Iterations |      Time | Objective |
|-----------|------------|-----------|-----------|
|         2 |        401 | 0.0970337 |  0.258898 |
|         4 |       5000 |   1.33258 |  0.253525 |
|        32 |        231 | 0.0963392 |  0.259886 |
|      1024 |        185 |   1.98786 |  0.266993 |
|     32768 |        157 |    205.02 |  0.259302 |

## Technical details

This tutorial is cached. It was last run on the following package versions.

``` julia
using Pkg
Pkg.status()
```

    Status `~/Repositories/Julia/ManoptExamples.jl/examples/Project.toml`
      [6e4b80f9] BenchmarkTools v1.5.0
      [336ed68f] CSV v0.10.15
      [35d6a980] ColorSchemes v3.27.1
    ⌅ [5ae59095] Colors v0.12.11
      [a93c6f00] DataFrames v1.7.0
      [7073ff75] IJulia v1.26.0
      [682c06a0] JSON v0.21.4
      [8ac3fa9e] LRUCache v1.6.1
      [d3d80556] LineSearches v7.3.0
      [af67fdf4] ManifoldDiff v0.3.13
      [1cead3c2] Manifolds v0.10.7
      [3362f125] ManifoldsBase v0.15.22
      [0fc0a36d] Manopt v0.5.3 `../../Manopt.jl`
      [5b8d5e80] ManoptExamples v0.1.10 `..`
      [51fcb6bd] NamedColors v0.2.2
      [91a5bcdd] Plots v1.40.9
    ⌃ [08abe8d2] PrettyTables v2.3.2
      [6099a3de] PythonCall v0.9.23
      [f468eda6] QuadraticModels v0.9.7
      [1e40b3f8] RipQP v0.6.4
    Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`

``` julia
using Dates
now()
```

    2024-11-28T00:40:27.330

## Literature

```@bibliography
Pages = ["RCBM-Median.md"]
Canonical=false
```
