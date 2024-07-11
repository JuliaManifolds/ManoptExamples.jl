A comparison of the RCBM with the PBA and the SGM for the Riemannian median
================
Hajg Jasa
6/27/24

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
using Random, LinearAlgebra, LRUCache
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
Random.seed!(33)
experiment_name = "RCBM-Median"
results_folder = joinpath(@__DIR__, experiment_name)
!isdir(results_folder) && mkdir(results_folder)

atol = 1e-8
N = 1000 # number of data points
spd_dims = [2, 5, 10, 15]
hn_sn_dims = [1, 2, 5, 10, 15]

# Generate a point that is at least `tol` close to the point `p` on `M`
function close_point(M, p, tol; retraction_method=Manifolds.default_retraction_method(M, typeof(p)))
    X = rand(M; vector_at = p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, retraction_method)
end
```

``` julia
# Objective and subdifferential
f(M, p, data) = sum(1 / length(data) * distance.(Ref(M), Ref(p), data))
domf(M, p, p0, diameter) = distance(M, p, p0) < diameter / 2 ? true : false
function ∂f(M, p, data, atol=atol)
    return sum(
        1 / length(data) *
        ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=atol),
    )
end
```

``` julia
rcbm_kwargs(diameter, domf, k_max) = [
    :diameter => diameter,
    :domain => domf,
    :k_max => k_max,
    :count => [:Cost, :SubGradient],
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :debug => [
        :Iteration,
        (:Cost, "F(p): %1.16f "),
        (:ξ, "ξ: %1.8f "),
        (:last_stepsize, "step size: %1.8f"),
        :Stop,
        1000,
        "\n",
    ],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
rcbm_bm_kwargs(diameter, domf, k_max) = [
    :diameter => diameter,
    :domain => domf,
    :k_max => k_max,
    :cache => (:LRU, [:Cost, :SubGradient], 50),
]
pba_kwargs = [
    :count => [:Cost, :SubGradient],
    :cache => (:LRU, [:Cost, :SubGradient], 50),
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
]
pba_bm_kwargs = [:cache => (:LRU, [:Cost, :SubGradient], 50),]
sgm_kwargs = [
    :count => [:Cost, :SubGradient],
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stepsize => DecreasingStepsize(1, 1, 0, 1, 0, :absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(1e-4) | StopAfterIteration(5000),
    :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
sgm_bm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stepsize => DecreasingStepsize(1, 1, 0, 1, 0, :absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(1e-4) | StopAfterIteration(5000),
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
k_max_hn = 0.0
diameter_hn = floatmax(Float64)
global A1, A2 = initialize_dataframes(
    results_folder,
    experiment_name,
    subexperiment_name,
    named_tuple_1,
    named_tuple_2
)

for n in hn_sn_dims

    M = Hyperbolic(Int(2^n))
    data = [rand(M) for _ in 1:N]
    dists = [distance(M, z, y) for z in data, y in data]
    p0 = data[minimum(Tuple(findmax(dists)[2]))]

    f_hn(M, p) = f(M, p, data)
    domf_hn(M, p) = domf(M, p, p0, diameter_hn)
    ∂f_hn(M, p) = ∂f(M, p, data, atol)

    # Optimization
    pba = proximal_bundle_method(M, f_hn, ∂f_hn, p0; pba_kwargs...)
    pba_result = get_solver_result(pba)
    pba_record = get_record(pba)

    rcbm = convex_bundle_method(M, f_hn, ∂f_hn, p0; rcbm_kwargs(diameter_hn, domf_hn, k_max_hn)...)
    rcbm_result = get_solver_result(rcbm)
    rcbm_record = get_record(rcbm)

    sgm = subgradient_method(M, f_hn, ∂f_hn, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        rcbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        rcbm_bm = @benchmark convex_bundle_method($M, $f_hn, $∂f_hn, $p0; rcbm_bm_kwargs($diameter_hn, $domf_hn, $k_max_hn)...)
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
|         2 |            9 | 0.00479385 |     1.07452 |          225 | 0.113286 |     1.07452 |
|         4 |            8 | 0.00464954 |     1.09164 |          235 | 0.120968 |     1.09164 |
|        32 |           13 |  0.0110309 |     1.10437 |          223 | 0.159239 |     1.10437 |
|      1024 |           16 |   0.250066 |     1.09989 |          240 |  3.53749 |     1.09989 |
|     32768 |           14 |    5.66661 |     1.07903 |          223 |  81.9645 |     1.07903 |

… Whereas the following table refers to the SGM

| Dimension | Iterations |       Time | Objective |
|-----------|------------|------------|-----------|
|         2 |         15 | 0.00681038 |   1.02872 |
|         4 |         18 | 0.00879092 |   1.08236 |
|        32 |         21 |  0.0168037 |   1.10419 |
|      1024 |         24 |   0.342026 |   1.09989 |
|     32768 |         19 |    6.39479 |   1.07883 |

## The Median on the Symmetric Positive Definite Matrix Space

``` julia
subexperiment_name = "SPD"
k_max_spd = 0.0
diameter_spd = floatmax(Float64)
global A1_SPD, A2_SPD = initialize_dataframes(
    results_folder,
    experiment_name,
    subexperiment_name,
    named_tuple_1,
    named_tuple_2
)

for n in spd_dims

    M = SymmetricPositiveDefinite(Int(n))
    data = [rand(M) for _ in 1:N]
    dists = [distance(M, z, y) for z in data, y in data]
    p0 = data[minimum(Tuple(findmax(dists)[2]))]

    f_spd(M, p) = f(M, p, data)
    domf_spd(M, p) = domf(M, p, p0, diameter_spd)
    ∂f_spd(M, p) = ∂f(M, p, data, atol)

    # Optimization
    pba = proximal_bundle_method(M, f_spd, ∂f_spd, p0; pba_kwargs...)
    pba_result = get_solver_result(pba)
    pba_record = get_record(pba)

    rcbm = convex_bundle_method(M, f_spd, ∂f_spd, p0; rcbm_kwargs(diameter_spd, domf_spd, k_max_spd)...)
    rcbm_result = get_solver_result(rcbm)
    rcbm_record = get_record(rcbm)

    sgm = subgradient_method(M, f_spd, ∂f_spd, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        rcbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        rcbm_bm = @benchmark convex_bundle_method($M, $f_spd, $∂f_spd, $p0; rcbm_bm_kwargs($diameter_spd, $domf_spd, $k_max_spd)...)
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
|         3 |           16 | 0.152925 |    0.254264 |           56 | 0.525707 |    0.254264 |
|        15 |           11 | 0.265103 |    0.433549 |           83 |   1.9075 |    0.433549 |
|        55 |           10 | 0.599664 |    0.624809 |           91 |  4.86849 |    0.624809 |
|       120 |            9 |  1.03983 |    0.769986 |          117 |  11.5079 |    0.769986 |

… Whereas the following table refers to the SGM

| Dimension | Iterations |    Time | Objective |
|-----------|------------|---------|-----------|
|         3 |       5000 | 41.8571 |  0.254264 |
|        15 |       1758 | 40.2008 |  0.433549 |
|        55 |        768 |  40.049 |  0.624809 |
|       120 |        429 | 41.8372 |  0.769986 |

## The Median on the Sphere

For the last experiment, note that a major difference here is that the sphere has constant positive sectional curvature equal to $1$. In this case, we lose the global convexity of the Riemannian distance and thus of the objective. Minimizers still exist, but they may, in general, be non-unique.

``` julia
subexperiment_name = "Sn"
k_max_sn = 1.0
diameter_sn = π / 4
global A1_Sn, A2_Sn = initialize_dataframes(
    results_folder,
    experiment_name,
    subexperiment_name,
    named_tuple_1,
    named_tuple_2
)

for n in hn_sn_dims

    M = Sphere(Int(2^n))
    north = [0.0 for _ in 1:manifold_dimension(M)]
    push!(north, 1.0)
    diameter = π / 4 #2 * π/7
    data = [close_point(M, north, diameter / 2) for _ in 1:n]
    p0 = data[1]

    f_sn(M, p) = f(M, p, data)
    domf_sn(M, p) = domf(M, p, p0, diameter_sn)
    ∂f_sn(M, p) = ∂f(M, p, data, atol)

    # Optimization
    pba = proximal_bundle_method(M, f_sn, ∂f_sn, p0; pba_kwargs...)
    pba_result = get_solver_result(pba)
    pba_record = get_record(pba)

    rcbm = convex_bundle_method(M, f_sn, ∂f_sn, p0; rcbm_kwargs(diameter_sn, domf_sn, k_max_sn)...)
    rcbm_result = get_solver_result(rcbm)
    rcbm_record = get_record(rcbm)

    sgm = subgradient_method(M, f_sn, ∂f_sn, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        rcbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        rcbm_bm = @benchmark convex_bundle_method($M, $f_sn, $∂f_sn, $p0; rcbm_bm_kwargs($diameter_sn, $domf_sn, $k_max_sn)...)
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

| Dimension | Iterations_1 |      Time_1 | Objective_1 | Iterations_2 |     Time_2 | Objective_2 |
|-----------|--------------|-------------|-------------|--------------|------------|-------------|
|         2 |            2 | 0.000121292 |         0.0 |            3 | 0.00012375 |         0.0 |
|         4 |         5000 |    0.740364 |    0.163604 |           30 | 0.00401738 |    0.163604 |
|        32 |           68 |  0.00849283 |    0.103703 |           37 | 0.00136038 |    0.103703 |
|      1024 |           65 |   0.0922398 |    0.119482 |           32 | 0.00716244 |    0.119482 |
|     32768 |           39 |      1.2733 |    0.176366 |           37 |   0.361279 |    0.176366 |

… Whereas the following table refers to the SGM

| Dimension | Iterations |        Time |   Objective |
|-----------|------------|-------------|-------------|
|         2 |       5000 |   0.0143786 | 9.98865e-15 |
|         4 |         30 | 0.000119917 |    0.163604 |
|        32 |       5000 |   0.0257619 |    0.103703 |
|      1024 |       5000 |    0.718539 |    0.119482 |
|     32768 |       5000 |     33.9769 |    0.176366 |

## Literature

```@bibliography
Pages = ["RCBM-Median.md"]
Canonical=false
```

