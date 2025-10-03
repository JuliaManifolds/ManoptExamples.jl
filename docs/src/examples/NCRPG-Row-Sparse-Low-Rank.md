# Row-sparse Low-rank Matrix Recovery
Paula John, Hajg Jasa
2025-10-01

## Introduction

In this example we use the Nonconvex Riemannian Proximal Gradient (NCRPG) method [BergmannJasaJohnPfeffer:2025:1](@cite) and compare it to the Riemannian Alternating Direction Method of Multipliers (RADMM) [JiaxiangShiqianTejes:2022:1](@cite).
This example reproduces the results from [BergmannJasaJohnPfeffer:2025:1](@cite), Section 6.3.
The numbers may vary slightly due to having run this notebook on a different machine.

``` julia
using PrettyTables
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots, LaTeXStrings
using Random, LinearAlgebra, LRUCache, Distributions
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
```

## The Problem

Let ``\mathcal M = \mathcal M_r`` be the manifold of rank ``r`` matrices.
Let ``g \colon \mathcal M \to \mathbb R`` be defined by

``` math
g(X) = \Vert\mathbb A (X) - y\Vert_2^2,
```

where ``\mathbb A \colon \mathbb R^{M \times N} \to \mathbb R^m`` is a linear measurement operator.

Let ``h \colon \mathcal M \to \mathbb R`` be defined by

``` math
h(X) = \mu \Vert X \Vert_{1, 2}
```

be the row sparsity-enforcing term given by the ``\ell_{1,2}``-norm, where ``\mu \ge 0`` is a regularization parameter.

We define our total objective function as ``f = g + h``.
The goal is to recover the (low-rank and row-sparse) signal ``X`` from as few measurements ``y`` as possible.

## Numerical Experiment

We initialize the experiment parameters, as well as some utility functions.

``` julia
# Set random seed for reproducibility
random_seed = 1520
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 180.0

atol = 1e-7
max_iters = 5000
c = 1e-4    # penalty parameter 
M = 500     # amount of rows
N = 100     # amount of columns
s = 10      # amount of non-zero rows
r_m_array = [(1, 300), (2, 500), (3, 700)] # (rank, number of measurements)
step_size = 0.25
init_step_size_bt = 2 * step_size
stop_NCRPG = atol
stop_RADMM = stop_NCRPG * step_size 
```

``` julia
function mean_error_zero_rows(Mr, X, zero_rows)
    M, N, r = Manifolds.get_parameter(Mr.size)
    err = 0.0
    for j in zero_rows
        err += norm(X.U[j, :] .* X.S)
    end 
    return err/length(zero_rows)
end
```

We define a function to generate the test data for the Sparse PCA problem.

``` julia
function generate_data(M, N, r, m, s)
    # Generate rank r matrix with s non-zero rows
    X = rand(M, r) * transpose(Matrix(qr(rand(N, r)).Q[:, 1:r]))
    smpl = sample(1:M, M - s , replace = false)
    X[smpl, :] .= 0.0

    # Normalize
    X = X/norm(X)

    # Generate measurement operator A and signal y
    A = rand(Normal(0, 1/sqrt(m)), m, M * N)
    y = A * vec(X)   
    return X, A, y, smpl
end
```

We implement the proximal operators for the ``\ell_{1, 2}``-norm on the fixed-rank manifold, following [BergmannJasaJohnPfeffer:2025:1](@cite) and [JiaxiangShiqianTejes:2022:1](@cite).

``` julia
# NCRPG 
function prox_norm12_global(Mr::FixedRankMatrices, prox_param, X::SVDMPoint; c = c) 
    λ = prox_param * c
    M, N, k = Manifolds.get_parameter(Mr.size)
    YU = zeros(M, k)
    for i in 1:M
        normx1_i = norm(X.U[i, :] .* X.S)
        if  normx1_i > λ
            YU[i, :] = ((normx1_i - λ)/normx1_i) * X.U[i, :]
        end
    end
    Z = SVDMPoint(YU * diagm(X.S))
    return SVDMPoint(Z.U, Z.S, Z.Vt * X.Vt)
end
#
#RADMM 
function prox_norm12_global(Mr::FixedRankMatrices, prox_param, X1::Matrix{Float64}; c = c) 
    λ = prox_param * c
    M, N, k = Manifolds.get_parameter(Mr.size)
    Y1 = zeros(M, N)
    for i in 1:M
        normx1_i = norm(X1[i, :])
        if  normx1_i > λ
            Y1[i, :] = ((normx1_i - λ)/normx1_i) * X1[i, :]
        end
    end
    return Y1
end
```

Next, we define the objective function, its gradient, and the proximal operator for the ``\ell_{1,2}``-norm on the fixed-rank manifold.

``` julia
# Objective(s), gradient, and proxes
function norm12(X::SVDMPoint)
    M = size(X.U)[1]
    sum([norm(X.U[i, :].*X.S) for i = 1:M])
end
function f_global(M::FixedRankMatrices, X::SVDMPoint, A, c, y, max_cost = 1e5) 
    X_vec = vec(embed(M, X))
    cost =  0.5 * (norm(A*X_vec - y))^2 + c * norm12(X)
    if cost >= max_cost
        return NaN
    else 
        return cost
    end 
end 
#
g_global(M::FixedRankMatrices, X::SVDMPoint, A, c, y) = 0.5*(norm(A*vec(embed(M, X)) - y))^2 
function grad_g_global(M::FixedRankMatrices, X::SVDMPoint, A, c, y) 
    X_mat = embed(M, X)
    return project(M, X, reshape(A'*(A*vec(X_mat) - y), size(X_mat)))
end 
h_global(M::FixedRankMatrices, X::SVDMPoint, A, c, y) = c * norm12(X)
```

We introduce an implementation of the RADMM method for the Row-sparse Low-rank Matrix Recovery problem on the oblique manifold, following [JiaxiangShiqianTejes:2022:1](@cite).

``` julia
"""
\argmin F(X) = 0.5||AX-y||^2 + c *||X||_{1,2}
f(X) = 0.5||AX - y||^2
g(X) = c * ||X||_{1,2}
L_{ρ,γ}(X, Z, Λ) = f(X) + g_γ(Z) + <Λ, X - Z> + ρ/2 * ||X - Z||^2
grad_X L_{ρ,γ}(X, Z, Λ) = project(A'(AX - y) + Λ + ρ(X - Z)) 

"""
function RADMM_blind_deconv(
    A,
    M, # rows
    N, # columns
    rank,
    c,  # penalty parameter 
    y, # signal
    prox_l12;
    η = 1e-1,
    ρ = 0.1,
    γ = 1e-7,
    stop = 1e-8, 
    max_iters  = 100,
    start = 0,
    record = false, 
    ϵ_spars = 1e-5,
    max_cost = 1e5
) 
    flag_succ = false
    Mr = FixedRankMatrices(M, N, rank)
    γρ = γ * ρ
    F(X) = 0.5 * norm(A*vec(X) - y)^2 + c * sum([norm(X[i, :]) for i=1:M])
    grad_augLagr(X::SVDMPoint, X_mat::Matrix, Λ::Matrix, Z::Matrix) = project(Mr, X, reshape(A'*(A*vec(X_mat) - y) + vec(Λ + ρ*(X_mat - Z)), (M, N)))

    if start == 0
        X = rand(Mr)
    else 
        X = start
    end 
    X_mat = embed(Mr, X)
    Z = X_mat
    Λ = zeros(M, N)
    it = -1 
    if !record 
        for i in 1:max_iters 
            descent_dir = -η * grad_augLagr(X, X_mat, Λ, Z)
            X = retract(Mr, X, descent_dir) 
            X_mat = embed(Mr, X)
            Y = prox_l12(Mr, (1 + γρ)/ρ , X_mat + 1/ρ * Λ; c = c)
            Z = (1/γ * Y + Λ + ρ * X_mat)/ (1/γ + ρ)
            Λ = Λ + ρ * (X_mat - Z)
            if (norm(embed(Mr, X, descent_dir)) < stop)
                flag_succ = true 
                it = i 
                break 
            end 
        end
        if it == -1
            it = max_iters 
        end 
        return X, flag_succ, it 
    else 
        Iterates = []
        for i in 1:max_iters 
            descent_dir = -η * grad_augLagr(X, X_mat, Λ, Z)
            X = retract(Mr, X, descent_dir) 
            push!(Iterates, X)
            X_mat = embed(Mr, X)
            Y = prox_l12(Mr, (1 + γρ)/ρ , X_mat + 1/ρ * Λ; c = c)
            Z = (1/γ * Y + Λ + ρ * X_mat)/ (1/γ + ρ)
            Λ = Λ + ρ * (X_mat - Z)
            if (norm(embed(Mr, X, descent_dir)) < stop)
                flag_succ = true 
                it = i 
                break 
            end 
            # if i%100 == 0
            #     println(i, "\t", norm(embed(Mr, X, descent_dir)))
            # end 
        end
        if it == -1 
            it = max_iters 
        end 
        return X, flag_succ, it, Iterates
    end 
end 
```

We set up some variables to collect the results of the experiments and initialize the dataframes

And run the experiments

``` julia
for (r, m) in r_m_array
    # Set random seed for reproducibility
    Random.seed!(random_seed)
    #
    # Define manifold
    Mr = FixedRankMatrices(M, N, r) #fixed rank manifold
    #
    # Generate rank r matrix with s non-zero rows
    Sol_mat, A, y, zero_rows = generate_data(M, N, r, m, s)
    Sol = SVDMPoint(Sol_mat, r)
    # Local starting point 
    Y = rand(Normal(0, 0.01/sqrt(r)), M, N)
    start = SVDMPoint(Sol_mat + Y, r)
    dist_start_sol = distance(Mr, Sol, start, OrthographicInverseRetraction())
    # Localize objectives
    f(Mr, X) = f_global(Mr, X, A, c, y)
    g(Mr, X) = g_global(Mr, X, A, c, y)
    grad_g(Mr, X) = grad_g_global(Mr, X, A, c, y)
    h(Mr, X) = h_global(Mr, X, A, c, y)
    prox_norm12(Mr, prox_param, X) = prox_norm12_global(Mr, prox_param, X; c = c)
    #
    # Optimization
    # NCRPG
    rec_NCRPG = proximal_gradient_method(Mr, f, g, grad_g, start; 
        prox_nonsmooth=prox_norm12,
        retraction_method=OrthographicRetraction(),
        inverse_retraction_method=OrthographicInverseRetraction(),
        stepsize = ConstantLength(step_size),
        record=[:Iteration, :Iterate],
        return_state=true,
        debug=[ 
            :Iteration,( "|Δp|: %1.9f |"),
            DebugChange(; inverse_retraction_method= OrthographicInverseRetraction()),
            (:Cost, " F(x): %1.11f | "), 
            "\n", 
            :Stop, 
            100
        ],
        stopping_criterion =  StopAfterIteration(max_iters )|StopWhenGradientMappingNormLess(stop_NCRPG)
    )
    bm_NCRPG = @benchmark proximal_gradient_method(``Mr, ``f, ``g, ``grad_g, ``start; 
        prox_nonsmooth=``prox_norm12,
        retraction_method=OrthographicRetraction(),
        inverse_retraction_method=OrthographicInverseRetraction(),
        stepsize = ConstantLength(``step_size),
        stopping_criterion = StopAfterIteration(``max_iters )|StopWhenGradientMappingNormLess(``stop_NCRPG)
    )
    it_NCRPG, res_NCRPG = get_record(rec_NCRPG)[end]
    # NCRPG with backtracking
    rec_NCRPG_bt = proximal_gradient_method(Mr, f, g, grad_g, start; 
        prox_nonsmooth=prox_norm12,
        retraction_method=OrthographicRetraction(),
        inverse_retraction_method=OrthographicInverseRetraction(),
        stepsize = ProximalGradientMethodBacktracking(;             
            strategy=:nonconvex, 
            initial_stepsize=init_step_size_bt
        ),
        record=[:Iteration, :Iterate],
        return_state=true,
        debug=[ 
            :Iteration,( "|Δp|: %1.9f |"),
            DebugChange(; inverse_retraction_method=OrthographicInverseRetraction()),
            (:Cost, " F(x): %1.11f | "), 
            "\n", 
            :Stop, 
            100
        ],
        stopping_criterion = StopAfterIteration(max_iters )|    StopWhenGradientMappingNormLess(stop_NCRPG)
    )
    bm_NCRPG_bt = @benchmark proximal_gradient_method(``Mr, ``f, ``g, ``grad_g, ``start; 
        prox_nonsmooth=``prox_norm12,
        retraction_method=OrthographicRetraction(),
        inverse_retraction_method=OrthographicInverseRetraction(),
        stepsize = ProximalGradientMethodBacktracking(; strategy=:nonconvex, initial_stepsize=``init_step_size_bt),
        stopping_criterion = StopAfterIteration(``max_iters )|  StopWhenGradientMappingNormLess(``stop_NCRPG)
    )
    it_NCRPG_bt, res_NCRPG_bt = get_record(rec_NCRPG_bt)[end]
    # RADMM
    res_RADMM, succ, it_RADMM = RADMM_blind_deconv(A, M, N, r, c, y, prox_norm12_global; 
        max_iters  = max_iters , 
        start = start, 
        η = step_size,  
        stop = stop_RADMM
    )  
    bm_RADMM = @benchmark RADMM_blind_deconv(``A, ``M, ``N, ``r, ``c, ``y, ``prox_norm12_global; 
        max_iters  = max_iters , 
        start = ``start, 
        η = ``step_size,  
        stop = stop_RADMM
    )  
    #
    # Collect results
    # Distances between the results
    dist_NCRPG_RADMM = distance(Mr, res_NCRPG, res_RADMM, OrthographicInverseRetraction())
    dist_NCRPG_bt_RADMM = distance(Mr, res_NCRPG_bt, res_RADMM, OrthographicInverseRetraction())
    dist_NCRPG_NCRPG_bt = distance(Mr, res_NCRPG, res_NCRPG_bt, OrthographicInverseRetraction())
    # Times
    time_RADMM    = median(bm_RADMM   ).time/1e9
    time_NCRPG    = median(bm_NCRPG   ).time/1e9
    time_NCRPG_bt = median(bm_NCRPG_bt).time/1e9
    # Errors
    error_RADMM    = distance(Mr, Sol, res_RADMM,    OrthographicInverseRetraction())
    error_NCRPG    = distance(Mr, Sol, res_NCRPG,    OrthographicInverseRetraction())
    error_NCRPG_bt = distance(Mr, Sol, res_NCRPG_bt, OrthographicInverseRetraction())
    # Mean zero row errors
    mean_zero_row_error_NCRPG    = mean_error_zero_rows(Mr, res_NCRPG, zero_rows    )
    mean_zero_row_error_NCRPG_bt = mean_error_zero_rows(Mr, res_NCRPG_bt, zero_rows )
    mean_zero_row_error_RADMM    = mean_error_zero_rows(Mr, res_RADMM, zero_rows    )
    #
    # Push results to dataframes
    push!(df_RADMM, 
        [
            M, N, m, r, s,
            step_size,
            time_RADMM, 
            error_RADMM,
            it_RADMM,
            mean_zero_row_error_RADMM
        ]
    )
    push!(df_NCRPG, 
        [
            M, N, m, r, s,
            step_size,
            time_NCRPG, 
            error_NCRPG,
            it_NCRPG,
            mean_zero_row_error_NCRPG
        ]
    )
    push!(df_NCRPG_bt, 
        [
            M, N, m, r, s,
            init_step_size_bt,
            time_NCRPG_bt, 
            error_NCRPG_bt,
            it_NCRPG_bt,
            mean_zero_row_error_NCRPG_bt
        ]
    )
    push!(df_distances, 
        [
            M, N, m, r, s,
            dist_NCRPG_NCRPG_bt,
            dist_NCRPG_RADMM,
            dist_NCRPG_bt_RADMM
        ]
    )
end
```

We export the results to CSV files

``` julia
CSV.write(joinpath(results_folder, "results-fixed-rank-RADMM.csv"), df_RADMM)
CSV.write(joinpath(results_folder, "results-fixed-rank-NCRPG.csv"), df_NCRPG)
CSV.write(joinpath(results_folder, "results-fixed-rank-NCRPG-bt.csv"), df_NCRPG_bt)
CSV.write(joinpath(results_folder, "results-fixed-rank-distances.csv"), df_distances )
```

We can take a look at how the algorithms compare to each other in their performance with the following tables.
The first table shows the performance RADMM.

| **M** | **N** | **m** | **r** | **s** | **stepsize** | **time (s)** | **error** | **iterations** | **mean\_zero\_row\_error** |
|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|--------------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------------------:|
|              500.0 |              100.0 |              300.0 |                1.0 |               10.0 |                      0.25 |               16.6644 |             0.00052812 |                      1431.0 |                              5.37441e-9 |
|              500.0 |              100.0 |              500.0 |                2.0 |               10.0 |                      0.25 |               19.1971 |            0.000725431 |                      1354.0 |                              6.09862e-9 |
|              500.0 |              100.0 |              700.0 |                3.0 |               10.0 |                      0.25 |               32.7179 |            0.000772805 |                      1414.0 |                               7.4266e-9 |

The next table shows the performance of NCRPG with a constant stepsize.

| **M** | **N** | **m** | **r** | **s** | **stepsize** | **time (s)** | **error** | **iterations** | **mean\_zero\_row\_error** |
|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|--------------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------------------:|
|              500.0 |              100.0 |              300.0 |                1.0 |               10.0 |                      0.25 |                10.475 |            0.000528145 |                      1049.0 |                                     0.0 |
|              500.0 |              100.0 |              500.0 |                2.0 |               10.0 |                      0.25 |               20.4298 |            0.000725859 |                      1047.0 |                             1.15293e-20 |
|              500.0 |              100.0 |              700.0 |                3.0 |               10.0 |                      0.25 |               22.6698 |            0.000775127 |                      1120.0 |                             4.22854e-20 |

The next table shows the performance of NCRPG with a backtracked stepsize.
In this case, the column "stepsize" indicates the initial stepsize for the backtracking procedure.

| **M** | **N** | **m** | **r** | **s** | **stepsize** | **time (s)** | **error** | **iterations** | **mean\_zero\_row\_error** |
|-------------------:|-------------------:|-------------------:|-------------------:|-------------------:|--------------------------:|----------------------:|-----------------------:|----------------------------:|----------------------------------------:|
|              500.0 |              100.0 |              300.0 |                1.0 |               10.0 |                       0.5 |               16.2641 |            0.000528144 |                       562.0 |                                     0.0 |
|              500.0 |              100.0 |              500.0 |                2.0 |               10.0 |                       0.5 |               31.1954 |            0.000725847 |                       604.0 |                             1.86904e-20 |
|              500.0 |              100.0 |              700.0 |                3.0 |               10.0 |                       0.5 |               3643.83 |            0.000778708 |                      5000.0 |                              6.3238e-20 |

Second, we look at the distances of the solutions found by each algorithm.

| **M** | **N** | **m** | **r** | **s** | **dist\_NCRPG\_NCRPG\_bt** | **dist\_NCRPG\_RADMM** | **dist\_NCRPG\_NCRPG\_bt** |
|------:|------:|------:|------:|------:|---------------------------:|-----------------------:|---------------------------:|
| 500.0 | 100.0 | 300.0 |   1.0 |  10.0 |                 1.08617e-8 |             5.59207e-7 |                 5.49924e-7 |
| 500.0 | 100.0 | 500.0 |   2.0 |  10.0 |                 2.18404e-8 |             7.53362e-7 |                 7.33125e-7 |
| 500.0 | 100.0 | 700.0 |   3.0 |  10.0 |                 1.38488e-5 |             2.39003e-5 |                 2.51751e-5 |

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
      [31c24e10] Distributions v0.25.120
      [7073ff75] IJulia v1.30.4
      [682c06a0] JSON v0.21.4
      [8ac3fa9e] LRUCache v1.6.2
      [b964fa9f] LaTeXStrings v1.4.0
      [d3d80556] LineSearches v7.4.0
      [ee78f7c6] Makie v0.24.6
      [af67fdf4] ManifoldDiff v0.4.4
      [1cead3c2] Manifolds v0.10.23
      [3362f125] ManifoldsBase v1.2.0
      [0fc0a36d] Manopt v0.5.23 `../../Manopt.jl`
      [5b8d5e80] ManoptExamples v0.1.15 `..`
      [51fcb6bd] NamedColors v0.2.3
      [91a5bcdd] Plots v1.41.1
      [08abe8d2] PrettyTables v3.0.11
      [6099a3de] PythonCall v0.9.28
      [f468eda6] QuadraticModels v0.9.14
      [1e40b3f8] RipQP v0.7.0
    Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated`

This tutorial was last rendered October 3, 2025, 15:22:53.

## Literature

```@bibliography
Pages = ["NCRPG-Sparse-PCA.md"]
Canonical=false
```
