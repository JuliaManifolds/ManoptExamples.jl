A comparison of the RCBM with the PBA, the SGM, and the CPPA for denoising a signal on the hyperbolic space
================
Hajg Jasa
01/11/2023

## Introduction

In this example we compare the Riemannian Convex Bundle Method (RCBM) [BergmannHerzogJasa:2024](@cite)
with the Proximal Bundle Algorithm, which was introduced in [HoseiniMonjeziNobakhtianPouryayevali:2021:1](@cite), and with the Subgradient Method (SGM), introduced in \[FerreiraOliveira:1998:1\], to solve the spectral Procrustes problem on $\mathrm{SO}(250)$.
This example reproduces the results from [BergmannHerzogJasa:2024](@cite), Section 5.

``` julia
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots
using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
```

## The Problem

Given two matrices $A, B \in \mathbb R^{n \times d}$ we aim to solve the orthogonal Procrustes problem

``` math
    {\arg\min}_{p \in \mathrm{SO}(d)}\ \Vert A - B \, p \Vert_2
    ,
```

where $\mathrm{SO}(d)$ is equipped with the standard bi-invariant metric, and where $\Vert \,\cdot\, \Vert_2$ denotes the spectral norm of a matrix, , its largest singular value.
We aim to find the best matrix $p \in \mathbb R^{d \times d}$ such that $p^\top p = \mathrm{id}$ is the identity matrix, or in other words $p$ is the best rotation.
Note that the spectral norm is convex in the Euclidean sense, but not geodesically convex on $\mathrm{SO}(d)$.
If we define the objective as

``` math
    f (p)
    = 
    \Vert A - B \, p \Vert_2
    ,
```

its subdifferential is given by

``` math
    \partial f(p) = \mathrm{proj}_p(-B^\top UV^\top)
```

where $U$ and $V$ are some left and right singular vectors, respectively, corresponding to the largest singular value of $A - B \, p$, and $\mathrm{proj}_p$ is the projection onto

``` math
    \mathcal T_p \mathrm{SO}(d)
    =
    \{
    A \in \mathbb R^{d,d} \, \vert \, pA^\top + Ap^\top = 0, \, \mathrm{trace}(p^{-1}A)=0
    \}
    .
```

## Numerical Experiment

We initialize the experiment parameters, as well as some utility functions.

``` julia
Random.seed!(33)
n = 1000
d = 250
A = rand(n, d)
B = randn(n, d)
tol = 1e-8
#
# Compute the orthogonal Procrustes minimizer given A and B
function orthogonal_procrustes(A, B)
    s =  svd((A'*B)')
    R = s.U* s.Vt
    return R
end
#
# Algorithm parameters
bundle_cap = 25
max_iters = 5000
δ = 0.#1e-2 # Update parameter for μ
μ = 50. # Initial proxiaml parameter for the proximal bundle method
k_max = 1/4
diameter = π/(3*√k_max)
#
# Manifolds and data
M = SpecialOrthogonal(d)
p0 = orthogonal_procrustes(A, B) #rand(M)
project!(M, p0, p0)
```

We now define objective and subdifferential (first the Euclidean one, then the projected one).

``` julia
f(M, p) = opnorm(A - B*p)
function ∂ₑf(M, p)
    cost_svd = svd(A - B*p)
    # Find all maxima in S – since S is sorted, these are the first n ones
    indices = [i for (i, v) in enumerate(cost_svd.S) if abs(v - cost_svd.S[1]) < eps()]
    ind = rand(indices)
    return -B'*(cost_svd.U[:,ind]*cost_svd.Vt[ind,:]')
end
rpb = Manifolds.RiemannianProjectionBackend(Manifolds.ExplicitEmbeddedBackend(M; gradient=∂ₑf))
∂f(M, p) = Manifolds.gradient(M, f, p, rpb)
domf(M, p) = distance(M, p, p0) < diameter/2 ? true : false
```

We introduce some keyword arguments for the solvers we will use in this experiment

``` julia
cbm_kwargs = [
    :bundle_cap => bundle_cap,
    :k_max => k_max,
    :domain => domf,
    :diameter => diameter,
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenLagrangeMultiplierLess(tol) | StopAfterIteration(max_iters),
    :debug => [
        :Iteration,
        (:Cost, "F(p): %1.16f "),
        (:ξ, "ξ: %1.8f "),
        (:ε, "ε: %1.8f "),
        (:last_stepsize, "step size: %1.8f"),
        :WarnBundle,
        :Stop,
        10,
        "\n",
    ],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
cbm_bm_kwargs = [
    :k_max => k_max,
    :domain => domf,
    :diameter => diameter,
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenLagrangeMultiplierLess(tol) | StopAfterIteration(max_iters),
]
pba_kwargs = [
    :bundle_size => bundle_cap,
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenLagrangeMultiplierLess(tol)|StopAfterIteration(max_iters),
    :debug =>[
        :Iteration,
        :Stop,
        (:Cost, "F(p): %1.16f "),
        (:ν, "ν: %1.16f "),
        (:c, "c: %1.16f "),
        (:μ, "μ: %1.8f "),
        :Stop,
        :WarnBundle,
        10,
        "\n",
    ],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
pba_bm_kwargs = [
    :cache =>(:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenLagrangeMultiplierLess(tol) |                                   StopAfterIteration(max_iters),
]
sgm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stepsize => DecreasingStepsize(1, 1, 0, 1, 0, :absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(√tol) | StopAfterIteration(max_iters),
    :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
    :record => [:Iteration, :Cost, :p_star],
    :return_state => true,
]
sgm_bm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stepsize => DecreasingStepsize(1, 1, 0, 1, 0, :absolute),
    :stopping_criterion => StopWhenSubgradientNormLess(√tol) |
                           StopAfterIteration(max_iters),
]
```

    3-element Vector{Pair{Symbol, Any}}:
                  :cache => (:LRU, [:Cost, :SubGradient], 50)
               :stepsize => DecreasingStepsize(; length=1.0,  factor=1.0,  subtrahend=0.0,  shift=0)
     :stopping_criterion => StopWhenAny with the Stopping Criteria
        Stop When _one_ of the following are fulfilled:
            |∂f| < 0.0001: not reached
            Max Iteration 5000: not reached
        Overall: not reached

We run the optimization algorithms…

``` julia
cbm = convex_bundle_method(M, f, ∂f, p0; cbm_kwargs...)
cbm_result = get_solver_result(cbm)
cbm_record = get_record(cbm)
#
pba = proximal_bundle_method(M, f, ∂f, p0; pba_kwargs...)
pba_result = get_solver_result(pba)
pba_record = get_record(pba)
#
sgm = subgradient_method(M, f, ∂f, p0; sgm_kwargs...)
sgm_result = get_solver_result(sgm)
sgm_record = get_record(sgm)
```

… And we benchmark their performance.

``` julia
if benchmarking
    pba_bm = @benchmark proximal_bundle_method($M, $f, $∂f, $p0; pba_bm_kwargs...)
    cbm_bm = @benchmark convex_bundle_method($M, $f, $∂f, $p0; cbm_bm_kwargs...)
    sgm_bm = @benchmark subgradient_method($M, $f, $∂f, $p0; sgm_bm_kwargs...)
    #
    experiments = ["RCBM", "PBA", "SGM"]
    records = [cbm_record, pba_record, sgm_record]
    results = [cbm_result, pba_result, sgm_result]
    times = [
        median(cbm_bm).time * 1e-9,
        median(pba_bm).time * 1e-9,
        median(sgm_bm).time * 1e-9,
    ]
    if show_plot
        global fig = plot(xscale=:log10)
    end
    #
    # Finalize - export costs
    if export_table
        for (time, record, result, experiment) in zip(times, records, results, experiments)
            C1 = [0.5 f(M, p0)]
            C = cat(first.(record), [r[2] for r in record]; dims=2)
            bm_data = vcat(C1, C)
            CSV.write(
                joinpath(results_folder, experiment_name * "_" * experiment * "-result.csv"),
                DataFrame(bm_data, :auto);
                header=["i", "cost"],
            )
            if show_plot
                plot!(fig, bm_data[:,1], bm_data[:,2]; label=experiment)
            end
        end
        D = cat(
            experiments,
            [maximum(first.(record)) for record in records],
            [t for t in times],
            [minimum([r[2] for r in record]) for record in records];
            dims=2,
        )
        CSV.write(
            joinpath(results_folder, experiment_name * "-comparisons.csv"),
            DataFrame(D, :auto);
            header=["Algorithm", "Iterations", "Time (s)", "Objective"],
        )
    end
end
```

    "/Users/hajgj/Repositories/Julia/ManoptExamples.jl/examples/Spectral-Procrustes/Spectral-Procrustes-comparisons.csv"

We can take a look at how the algorithms compare to each other in their performance with the following table…

``` julia
export_table && CSV.read(joinpath(experiment_name, experiment_name * "-comparisons.csv"), DataFrame; delim = ",")
```

… and this cost versus iterations plot

``` julia
show_plot && fig
```

![](Spectral-Procrustes_files/figure-commonmark/cell-10-output-1.svg)

## Introduction

In this example we compare the Riemannian Convex Bundle Method (RCBM) [BergmannHerzogJasa:2024](@cite)
with the Proximal Bundle Algorithm, which was introduced in [HoseiniMonjeziNobakhtianPouryayevali:2021:1](@cite), and with the Subgradient Method (SGM), introduced in \[FerreiraOliveira:1998:1\], to find the Riemannian median.
This example reproduces the results from [BergmannHerzogJasa:2024](@cite), Section 5.

``` {julia}
#| echo: false
#| code-fold: true
#| output: false
using Pkg;
cd(@__DIR__)
Pkg.activate("."); # for reproducibility use the local tutorial environment.
Pkg.develop(path="../") # a trick to work on the local dev version
benchmarking = true
export_table = true
plot = true
```

``` {julia}
#| output: false
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

``` {julia}
#| output: false
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

``` {julia}
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

``` {julia}
cbm_kwargs(diameter, domf, k_max) = [
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
cbm_bm_kwargs(diameter, domf, k_max) = [
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

``` {julia}
function initialize_dataframes(results_folder, experiment_name, subexperiment_name)
    A1 = DataFrame(;
            a="Dimension",
            b="Iterations",
            c="Time (s)",
            d="Objective",
            e="Iterations",
            f="Time (s)",
            g="Objective",
        )
    CSV.write(
        joinpath(
            results_folder,
            experiment_name * "_$subexperiment_name" * "-Comparisons-Convex-Prox.csv",
        ),
        A1;
        header=false,
    )
    A2 = DataFrame(;
        a="Dimension",
        b="Iterations",
        c="Time (s)",
        d="Objective",
    )
    CSV.write(
        joinpath(
            results_folder,
            experiment_name * "_$subexperiment_name" * "-Comparisons-Subgrad.csv",
        ),
        A2;
        header=false,
    )
end
```

``` {julia}
function write_dataframes(M, records, times, results_folder, experiment_name, subexperiment_name)
    B1 = DataFrame(;
        a=manifold_dimension(M),
        b=maximum(first.(records[1])),
        c=times[1],
        d=minimum([r[2] for r in records[1]]),
        e=maximum(first.(records[2])),
        f=times[2],
        g=minimum([r[2] for r in records[2]]),
    )
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
    B2 = DataFrame(;
        a=manifold_dimension(M),
        b=maximum(first.(records[3])),
        c=times[3],
        d=minimum([r[2] for r in records[3]]),
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

``` {julia}
subexperiment_name = "Hn"
k_max_hn = 0.0
diameter_hn = floatmax(Float64)
initialize_dataframes(results_folder, experiment_name, subexperiment_name)
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

    cbm = convex_bundle_method(M, f_hn, ∂f_hn, p0; cbm_kwargs(diameter_hn, domf_hn, k_max_hn)...)
    cbm_result = get_solver_result(cbm)
    cbm_record = get_record(cbm)

    sgm = subgradient_method(M, f_hn, ∂f_hn, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        cbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        cbm_bm = @benchmark convex_bundle_method($M, $f, $∂f, $p0; cbm_bm_kwargs($diameter_hn, $domf_hn, $k_max_hn)...)
        pba_bm = @benchmark proximal_bundle_method($M, $f, $∂f, $p0; pba_bm_kwargs...)
        sgm_bm = @benchmark subgradient_method($M, $f, $∂f, $p0; sgm_bm_kwargs...)
        times = [
            median(cbm_bm).time * 1e-9,
            median(pba_bm).time * 1e-9,
            median(sgm_bm).time * 1e-9,
        ]
        (export_table) && (write_dataframes(M, records, times, results_folder, experiment_name, subexperiment_name))
    end
end
```

## The Median on the Symmetric Positive Definite Matrix Space

``` {julia}
subexperiment_name = "SPD"
k_max_spd = 0.0
diameter_spd = floatmax(Float64)
initialize_dataframes(results_folder, experiment_name, subexperiment_name)
for n in spd_dims

    M = SymmetricPositiveDefinite(Int(2^n))
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

    cbm = convex_bundle_method(M, f_spd, ∂f_spd, p0; cbm_kwargs(diameter_spd, domf_spd, k_max_spd)...)
    cbm_result = get_solver_result(cbm)
    cbm_record = get_record(cbm)

    sgm = subgradient_method(M, f_spd, ∂f_spd, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        cbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        cbm_bm = @benchmark convex_bundle_method($M, $f, $∂f, $p0; cbm_bm_kwargs($diameter_spd, $domf_spd, $k_max_spd)...)
        pba_bm = @benchmark proximal_bundle_method($M, $f, $∂f, $p0; pba_bm_kwargs...)
        sgm_bm = @benchmark subgradient_method($M, $f, $∂f, $p0; sgm_bm_kwargs...)
        times = [
            median(cbm_bm).time * 1e-9,
            median(pba_bm).time * 1e-9,
            median(sgm_bm).time * 1e-9,
        ]
        (export_table) && (write_dataframes(M, records, times, results_folder, experiment_name, subexperiment_name)) 
    end
end
```

## The Median on the Sphere

``` {julia}
subexperiment_name = "Sn"
k_max_sn = 1.0
diameter_sn = π / 4
initialize_dataframes(results_folder, experiment_name, subexperiment_name)
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

    cbm = convex_bundle_method(M, f_sn, ∂f_sn, p0; cbm_kwargs(diameter_sn, domf_sn, k_max_sn)...)
    cbm_result = get_solver_result(cbm)
    cbm_record = get_record(cbm)

    sgm = subgradient_method(M, f_sn, ∂f_sn, p0; sgm_kwargs...)
    sgm_result = get_solver_result(sgm)
    sgm_record = get_record(sgm)

    records = [
        cbm_record,
        pba_record,
        sgm_record,
    ]

    if benchmarking
        cbm_bm = @benchmark convex_bundle_method($M, $f, $∂f, $p0; cbm_bm_kwargs($diameter_sn, $domf_sn, $k_max_sn)...)
        pba_bm = @benchmark proximal_bundle_method($M, $f, $∂f, $p0; pba_bm_kwargs...)
        sgm_bm = @benchmark subgradient_method($M, $f, $∂f, $p0; sgm_bm_kwargs...)
        times = [
            median(cbm_bm).time * 1e-9,
            median(pba_bm).time * 1e-9,
            median(sgm_bm).time * 1e-9,
        ]
        (export_table) && (write_dataframes(M, records, times, results_folder, experiment_name, subexperiment_name))
    end
end
```

## Introduction

In this example we compare the Riemannian Convex Bundle Method (RCBM) [BergmannHerzogJasa:2024](@cite)
with the Proximal Bundle Algorithm, which was introduced in [HoseiniMonjeziNobakhtianPouryayevali:2021:1](@cite), and with the Subgradient Method (SGM), introduced in [FerreiraOliveira:1998:1](@cite), to denoise an artificial signal on the Hyperbolic space $\mathcal H^2$.
This example reproduces the results from [BergmannHerzogJasa:2024](@cite), Section 5.2.

``` {julia}
#| echo: false
#| code-fold: true
#| output: false
using Pkg;
cd(@__DIR__)
Pkg.activate("."); # for reproducibility use the local tutorial environment.

Pkg.develop(path="../") # a trick to work on the local dev version

export_orig = true
export_table = true
export_result = true
benchmarking = true

experiment_name = "H2-Signal-TV"
results_folder = joinpath(@__DIR__, experiment_name)
!isdir(results_folder) && mkdir(results_folder)
```

``` {julia}
#| output: false
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots
using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
```

## The Problem

Let $\mathcal M = \mathcal H^2$ be the $2$-dimensional hyperbolic space and let $p, q \in \mathcal M^n$ be two manifold-valued signals, for $n \in \mathbb N$.
Let $f \colon \mathcal M \to \mathbb R$ be defined by

``` math
    f_q (p)
    = 
    \frac{1}{n}
    \{
    \frac{1}{2} \sum_{i = 1}^n \mathrm{dist}(p_i, q_i)^2
    +
    \alpha \operatorname{TV}(p)
    \}
    ,
```

where $\operatorname{TV}(p)$, is the total variation term given by

``` math
    \operatorname{TV}(p)
    =
    \sum_{i = 1}^{n-1} \mathrm{dist}(p_i, p_{i+1})
    .
```

## Numerical Experiment

We initialize the experiment parameters, as well as some utility functions.

``` {julia}
#| output: false
Random.seed!(33)
n = 496 # (this is so that n equals the actual length of the artificial signal)
σ = 0.1 # Noise parameter
α = 0.05 # TV parameter
atol = 1e-8
k_max = 0.0
max_iters = 15000
#
# Colors
data_color = RGBA{Float64}(colorant"#BBBBBB")
noise_color = RGBA{Float64}(colorant"#33BBEE") # Tol Vibrant Teal
result_color = RGBA{Float64}(colorant"#EE7733") # Tol Vibrant Orange
```

``` {julia}
#| output: false
function artificial_H2_signal(
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
    y = map(z -> Manifolds._hyperbolize(Hyperbolic(2), z), y)
    data = []
    geodesics = []
    l = Int(round(pts * T / (2 * (b - a))))
    for i in 1:2:(length(y) - 1)
        append!(
            data,
            shortest_geodesic(Hyperbolic(2), y[i], y[i + 1], range(0.0, 1.0; length=l)),
        )
        if i + 2 ≤ length(y) - 1
            append!(
                geodesics,
                shortest_geodesic(Hyperbolic(2), y[i], y[i + 1], range(0.0, 1.0; length=l)),
            )
            append!(
                geodesics,
                shortest_geodesic(
                    Hyperbolic(2), y[i + 1], y[i + 2], range(0.0, 1.0; length=l)
                ),
            )
        end
    end
    #! In order to have length(data) ∝ pts, we need typeof(l) == Int and mod(pts, l) == 0.
    if pts != length(data)
        @warn "The length of the output signal will differ from the input number of points."
    end
    return data, geodesics
end
function matrixify_Poincare_ball(input)
    input_x = []
    input_y = []
    for p in input
        push!(input_x, p.value[1])
        push!(input_y, p.value[2])
    end
    return hcat(input_x, input_y)
end
```

We now fix the data for the experiment…

``` {julia}
#| output: false
H = Hyperbolic(2)
data, geodesics = artificial_H2_signal(n; a=-6.0, b=6.0, T=3)
Hn = PowerManifold(H, NestedPowerRepresentation(), length(data))
noise = map(p -> exp(H, p, rand(H; vector_at=p, σ=σ)), data)
p0 = noise
diameter = floatmax(Float64)
```

… As well as objective, subdifferential, and proximal map.

``` {julia}
#| output: false
function f(M, p)
    return 1 / length(data) *
           (1 / 2 * distance(M, data, p)^2 + α * ManoptExamples.Total_Variation(M, p))
end
domf(M, p) = distance(M, p, p0) < diameter / 2 ? true : false
function ∂f(M, p)
    return 1 / length(data) * (
        ManifoldDiff.grad_distance(M, data, p) +
        α * ManoptExamples.subgrad_Total_Variation(M, p; atol=atol)
    )
end
proxes = (
    (M, λ, p) -> ManifoldDiff.prox_distance(M, λ, data, p, 2),
    (M, λ, p) -> ManoptExamples.prox_Total_Variation(M, α * λ, p),
)
```

We can now plot the initial setting.

``` {julia}
global ball_scene = plot()
if export_orig
    ball_data = convert.(PoincareBallPoint, data)
    ball_noise = convert.(PoincareBallPoint, noise)
    ball_geodesics = convert.(PoincareBallPoint, geodesics)
    plot!(ball_scene, H, ball_data; geodesic_interpolation=100, label="Geodesics")
    plot!(
        ball_scene,
        H,
        ball_data;
        markercolor=data_color,
        markerstrokecolor=data_color,
        label="Data",
    )
    plot!(
        ball_scene,
        H,
        ball_noise;
        markercolor=noise_color,
        markerstrokecolor=noise_color,
        label="Noise",
    )
    display(ball_scene)
    matrix_data = matrixify_Poincare_ball(ball_data)
    matrix_noise = matrixify_Poincare_ball(ball_noise)
    matrix_geodesics = matrixify_Poincare_ball(ball_geodesics)
    CSV.write(
        joinpath(results_folder, experiment_name * "-data.csv"),
        DataFrame(matrix_data, :auto);
        header=["x", "y"],
    )
    CSV.write(
        joinpath(results_folder, experiment_name * "-noise.csv"),
        DataFrame(matrix_noise, :auto);
        header=["x", "y"],
    )
    CSV.write(
        joinpath(results_folder, experiment_name * "-geodesics.csv"),
        DataFrame(matrix_geodesics, :auto);
        header=["x", "y"],
    )
end
```

We introduce some keyword arguments for the solvers we will use in this experiment

``` {julia}
cbm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :diameter => diameter,
    :domain => domf,
    :k_max => k_max,
    :debug => [
        :Iteration,
        (:Cost, "F(p): %1.8f "),
        (:ξ, "ξ: %1.16f "),
        (:ε, "ε: %1.16f "),
        :WarnBundle,
        :Stop,
        1000,
        "\n",
        ],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
cbm_bm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :diameter => diameter,
    :domain => domf,
    :k_max => k_max,
]
pba_kwargs = [
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
    :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iters),
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
pba_bm_kwargs = [
    :cache =>(:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenLagrangeMultiplierLess(atol) |                                   StopAfterIteration(max_iters),
]
sgm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenSubgradientNormLess(√atol) | StopAfterIteration(max_iters),
    :debug => [:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
sgm_bm_kwargs = [
    :cache => (:LRU, [:Cost, :SubGradient], 50),
    :stopping_criterion => StopWhenSubgradientNormLess(√atol) |
                           StopAfterIteration(max_iters),
]
cppa_kwargs = [
    :stopping_criterion => StopWhenAny(
        StopAfterIteration(max_iters), StopWhenChangeLess(atol)
    ),
    :debug => [
        :Iteration,
        " | ",
        DebugProximalParameter(),
        " | ",
        (:Cost, "F(p): %1.16f "),
        " | ",
        :Change,
        "\n",
        1000,
        :Stop,
    ],
    :record => [:Iteration, :Cost, :Iterate],
    :return_state => true,
]
cppa_bm_kwargs = [
    :stopping_criterion => StopWhenAny(
        StopAfterIteration(max_iters), StopWhenChangeLess(atol)
    ),
]
```

Finally, we run the optimization algorithms…

``` {julia}
#| output: false
cbm = convex_bundle_method(Hn, f, ∂f, p0; cbm_kwargs...)
cbm_result = get_solver_result(cbm)
cbm_record = get_record(cbm)
#
pba = proximal_bundle_method(Hn, f, ∂f, p0; pba_kwargs...)
pba_result = get_solver_result(pba)
pba_record = get_record(pba)
#
sgm = subgradient_method(Hn, f, ∂f, p0; sgm_kwargs...)
sgm_result = get_solver_result(sgm)
sgm_record = get_record(sgm)
#
cppa = cyclic_proximal_point(Hn, f, proxes, p0; cppa_kwargs...)
cppa_result = get_solver_result(cppa)
cppa_record = get_record(cppa)
```

… And we benchmark their performance.

``` {julia}
if benchmarking
    pba_bm = @benchmark proximal_bundle_method($Hn, $f, $∂f, $p0; pba_bm_kwargs...)
    cbm_bm = @benchmark convex_bundle_method($Hn, $f, $∂f, $p0; cbm_bm_kwargs...)
    sgm_bm = @benchmark subgradient_method($Hn, $f, $∂f, $p0; sgm_bm_kwargs...)
    cppa_bm = @benchmark cyclic_proximal_point($Hn, $f, $proxes, $p0; cppa_bm_kwargs...)
    #
    experiments = ["CBM", "PBA", "SGM", "CPPA"]
    records = [cbm_record, pba_record, sgm_record, cppa_record]
    results = [cbm_result, pba_result, sgm_result, cppa_result]
    times = [
        median(cbm_bm).time * 1e-9,
        median(pba_bm).time * 1e-9,
        median(sgm_bm).time * 1e-9,
        median(cppa_bm).time * 1e-9,
    ]
    #
    # Finalize - export costs
    if export_table
        for (time, record, result, experiment) in zip(times, records, results, experiments)
            A = cat(first.(record), [r[2] for r in record]; dims=2)
            CSV.write(
                joinpath(results_folder, experiment_name * "_" * experiment * "-result.csv"),
                DataFrame(A, :auto);
                header=["i", "cost"],
            )
        end
        B = cat(
            experiments,
            [maximum(first.(record)) for record in records],
            [t for t in times],
            [minimum([r[2] for r in record]) for record in records],
            [distance(Hn, data, result) / length(data) for result in results];
            dims=2,
        )
        CSV.write(
            joinpath(results_folder, experiment_name * "-comparisons.csv"),
            DataFrame(B, :auto);
            header=["Algorithm", "Iterations", "Time (s)", "Objective", "Error"],
        )
    end
end
```

Lastly, we plot the results.

``` {julia}
if export_result
    # Convert hyperboloid points to Poincaré ball points
    ball_b = convert.(PoincareBallPoint, cbm_result)
    ball_p = convert.(PoincareBallPoint, pba_result)
    ball_s = convert.(PoincareBallPoint, sgm_result)
    ball_c = convert.(PoincareBallPoint, cppa_result)
    #
    # Plot results
    plot!(
        ball_scene,
        H,
        ball_b;
        markercolor=result_color,
        markerstrokecolor=result_color,
        label="Convex Bundle Method",
    )
    #
    # Suppress some plots for clarity
    # plot!(ball_scene, H, ball_p; label="Proximal Bundle Method")
    # plot!(ball_scene, H, ball_s; label="Subgradient Method")
    # plot!(ball_scene, H, ball_c; label="CPPA")
    display(ball_scene)
    #
    # Write csv files
    matrix_b = matrixify_Poincare_ball(ball_b)
    CSV.write(
        joinpath(results_folder, experiment_name * "-bundle_optimum.csv"),
        DataFrame(matrix_b, :auto);
        header=["x", "y"],
    )
end
```
