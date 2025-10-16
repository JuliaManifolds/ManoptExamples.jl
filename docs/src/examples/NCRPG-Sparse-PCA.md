# Sparse PCA
Paula John, Hajg Jasa
2025-10-01

## Introduction

In this example we use the Nonconvex Riemannian Proximal Gradient (NCRPG) method [BergmannJasaJohnPfeffer:2025:1](@cite) and compare it to the Riemannian Proximal Gradient (RPG) method [HuangWei:2021:1](@cite).
This example reproduces the results from [BergmannJasaJohnPfeffer:2025:1](@cite), Section 6.1.
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

Let `\mathcal M = \mathrm{OB}(n,r)` be the oblique manifold, i.e., the set of `n \times r` matrices with unit-norm columns.
Let `g \colon \mathcal M \to \mathbb R` be defined by

``` math
g(X) = \frac{1}{2} \Vert X^\top A^\top A X - D^2 \Vert^2,
```

where `A \in \mathbb R^{m \times n}` is a data matrix, `D = \mathrm{diag}(d_1, \ldots, d_r)` is a diagonal matrix containing the top `r` singular values of `A`, and `\Vert \cdot \Vert` is the Frobenius norm.

Let `h \colon \mathcal M \to \mathbb R` be defined by

``` math
h(X) = \mu \Vert X \Vert_1
```

be the sparsity-enforcing term given by the `\ell_1`-norm, where `\mu \ge 0` is a regularization parameter.

We define our total objective function as `f = g + h`.
The goal is to find the minimizer of `f` on `\mathcal M`, which is heuristically the point that diagonalizes `A^\top A` as much as possible while being sparse.

## Numerical Experiment

We initialize the experiment parameters, as well as some utility functions.

``` julia
# Set random seed for reproducibility
random_seed = 1520
Random.seed!(random_seed)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0
m_tests = 10 # number of tests for each parameter setting
means = 20 # number of means to compute

atol = 1e-7
max_iters = 100000
n_p_array = [(100,5), (200,5), (300, 5)]
μs = [t for t in [0.1, 0.5, 1.0]]
```

We define a function to generate the test data for the Sparse PCA problem.

``` julia
function gen_test_data_SPCA(n, m, p)
    A = rand(Normal(0, 1.0), (m, n))
    for i in 1:n
        A[:, i] = A[:, i] .- mean(A[:, i])
        A[:, i] = A[:, i] / std(A[:, i])
    end
    svdA = svd(A)
    Vt = svdA.Vt
    PCs = Vt[:, 1:p]
    d = svdA.S[1:p]
    return A, PCs, d
end
```

We define the proximal operator for the `\ell_1`-norm on the oblique manifold, following [BergmannJasaJohnPfeffer:2025:1](@cite).

``` julia
# Returns prox_{μ||.||_1}(M,x) on the Oblique Manifold OB(n,p) with respect to riemannian distance
function prox_l1_OB(n, p, μ; tol = 1e-10, max_iters = 10)
    return function prox_l1_OB_μ(M, λ, X)
        μλ = μ * λ
        prox_X = Array{Float64}(undef, n, p)
        for k in 1:p
            x = X[:, k]
            t = μλ
            px_t = X[:, k]
            for _ in 1:max_iters
                t_old = t

                z = abs.(x) .- t
                prox_Rn_t = (z .> 0) .* sign.(x) .* z

                px_t = prox_Rn_t/norm(prox_Rn_t)
                xpx_t = x'px_t
                if xpx_t < 1
                    t = μλ * sqrt(1-(xpx_t)^2)/acos(xpx_t)
                else
                    px_t = x
                    prox_X[:, k] = x
                    break
                end
                if abs(t - t_old) < tol
                    prox_X[:, k] = px_t
                    break
                end
            end
            prox_X[:, k] = px_t
        end
        return prox_X
    end
end
```

``` julia
# Objective, gradient, and proxes
g_(M, X, H, D) = 0.5 * norm(X'H * X - D)^2
function grad_g_(M, X, H, D)
    HX = H*X
    return project(M, X, 2*HX*(X'HX - D))
end
h_(M, X, μ) = μ * norm(X, 1)
f_(M, X, H, D, μ) = 0.5 * norm(X'H * X - D)^2 + μ * norm(X, 1)
```

We introduce an implementation of the RPG method for the Sparse PCA problem on the oblique manifold, following [HuangWei:2021:1](@cite).

``` julia
# Implementation of the proximal operator for the ℓ1-norm on the Oblique manifold
function RPG_prox_OB(S, X, grad_fX, λ, L, n, p; max_iters  = 10, tol=1e-10)
    λ̃ = λ/L
    d = 0
    for k in 1:p
        x = X[:,k]
        ξ_x = 1/L * grad_fX[:,k]

        neg∇h = (x-ξ_x)/λ̃
        i_max = argmax(abs.(neg∇h))
        if abs(neg∇h[i_max]) <= 1.0
            y = sign(neg∇h[i_max])*(1:n .== i_max)
        else
            z = abs.(neg∇h) .- 1.0
            Act_set = z .> 0
            y = Act_set .* sign.(neg∇h) .* z
            y = y/norm(y)
        end
        for j in 1:max_iters
            xty = x'y
            if xty >= 1
                sy = -1
                ty = 1
            else
                ξty = ξ_x'y
                acosxty = acos(xty)
                α = 1-xty^2
                sy = - acosxty/sqrt(α) - ξty/α + acosxty * ξty * xty / sqrt(α^3)
                ty = acosxty/sqrt(α)
            end
            neg∇h = -(sy*x+ty*ξ_x)/λ̃

            i_max = argmax(abs.(neg∇h))
            if abs(neg∇h[i_max]) <= 1.0
                y_new = sign(neg∇h[i_max])*(1:n .== i_max)
            else
                z = abs.(neg∇h) .- 1.0
                Act_set = z .> 0
                y_new = Act_set .* sign.(neg∇h) .* z
                y_new = y_new/norm(y_new)
            end

            if max(abs(xty-x'y_new), abs(ξ_x'y-ξ_x'y_new)) < tol
                break
            end
            y = y_new
        end

        d += distance(S, x,y)^2
        X[:,k] = y
    end
    return sqrt(d)
end
#
# RPG implementation for Sparse PCA on the Oblique manifold
function RPG_SPCA_OB(M, H, D, μ, L, start, prox_fun; max_iters  = 1000, stop = 1e-8, record = false)
    n, p = size(start)
    S = Sphere(n-1)
    cost_fun(M,X) =  0.5*norm(X'H*X-D)^2 + μ*norm(X,1)
    function grad_f(M,X, D=D)
        HX = H*X
        return project(M, X, 2*HX*(X'HX-D))
    end
    X = copy(start)
    if !record
        for i in 1:max_iters
            change = prox_fun(S, X, grad_f(M,X,D), μ, L, n, p)
            if L*change < stop
                return X, i
            end
        end
        return X, max_iters
    else
        Iterates = []
        for i in 1:max_iters
            change = prox_fun(S, X, grad_f(M,X,D), μ, L, n, p)
            push!(Iterates, copy(X))
            if L*change < stop
                return Iterates, i
            end
        end
        return Iterates, max_iters
    end
end
```

We set up some variables to collect the results of the experiments and initialize the dataframes

And run the experiments

``` julia
for (n, p) in n_p_array
    # Define manifold
    OB = Oblique(n, p)
    for m in 1:m_tests
        # Construct problem
        A, PCs, d = gen_test_data_SPCA(n, means, p)
        H = A'A / norm(A'A) * 10
        D = diagm(svd(H).S[1:p])
        L = 2 * tr(H)

        for (c, μ) in enumerate(μs)
            # Localize functions
            g(M, X) = g_(M, X, H, D)
            grad_g(M, X) = grad_g_(M, X, H, D)
            h(M, X) = h_(M, X, μ)
            prox_norm1_NCRPG = prox_l1_OB(n, p, μ)
            f(M, X) = f_(M, X, H, D, μ)
            #
            # Parameters
            step_size = 1/L
            init_step_size_bt = 10 * step_size
            stop_step_size_bt = atol
            stop_RPG = atol
            stop_NCRPG = atol
            stop_NCRPG_bt = atol
            #
            # Fix starting point
            start = rand(OB)
            #
            # Optimization
            # NCRPG
            rec_NCRPG = proximal_gradient_method(OB, f, g, grad_g, start;
                prox_nonsmooth = prox_norm1_NCRPG,
                stepsize = ConstantLength(step_size),
                record = [:Iteration, :Iterate],
                return_state = true,
                stopping_criterion = StopAfterIteration(max_iters)| StopWhenGradientMappingNormLess(stop_NCRPG)
            )
            # Benchmark NCRPG
            bm_NCRPG = @benchmark proximal_gradient_method($OB, $f, $g, $grad_g, $start;
                prox_nonsmooth = $prox_norm1_NCRPG,
                stepsize = ConstantLength($step_size),
                stopping_criterion = StopAfterIteration($max_iters)| StopWhenGradientMappingNormLess($stop_NCRPG)
            )
            # NCRPG with backtracking
            rec_NCRPG_bt = proximal_gradient_method(OB, f, g, grad_g, start;
                prox_nonsmooth = prox_norm1_NCRPG,
                stepsize = ProximalGradientMethodBacktracking(;
                    strategy = :nonconvex,
                    initial_stepsize = init_step_size_bt,
                    stop_when_stepsize_less = stop_step_size_bt
                ),
                record = [:Iteration, :Iterate],
                return_state = true,
                stopping_criterion = StopAfterIteration(max_iters)| StopWhenGradientMappingNormLess(stop_NCRPG_bt)
            )
            # Benchmark NCRPG with backtracking
            bm_NCRPG_bt = @benchmark proximal_gradient_method($OB, $f, $g, $grad_g, $start;
                prox_nonsmooth = $prox_norm1_NCRPG,
                stepsize = ProximalGradientMethodBacktracking(;
                    strategy = :nonconvex,
                    initial_stepsize = $init_step_size_bt,
                    stop_when_stepsize_less = $stop_step_size_bt
                ),
                stopping_criterion = StopAfterIteration($max_iters)| StopWhenGradientMappingNormLess($stop_NCRPG_bt)
            )
            # RPG
            Iterates_RPG, it_RPG = RPG_SPCA_OB(OB, H, D, μ, L, start, RPG_prox_OB;
                max_iters = max_iters,
                stop = stop_RPG,
                record = true
            )
            bm_RPG = @benchmark RPG_SPCA_OB($OB, $H, $D, $μ, $L, $start, $RPG_prox_OB;
                max_iters = $max_iters,
                stop = $stop_RPG
            )
            #
            # Collect test results
            Iterates_NCRPG  = get_record(rec_NCRPG, :Iteration, :Iterate)
            res_NCRPG       = Iterates_NCRPG[end]
            time_NCRPG      = time(median(bm_NCRPG))/1e9
            obj_NCRPG       = f(OB, res_NCRPG)
            spar_NCRPG      = sum(abs.(res_NCRPG).< 1e-8)/n/p
            it_NCRPG        = length(Iterates_NCRPG)
            orth_NCRPG      = norm(res_NCRPG'*res_NCRPG - I(p))
            # NCRPG with backtracking
            Iterates_NCRPG_bt  = get_record(rec_NCRPG_bt, :Iteration, :Iterate)
            res_NCRPG_bt       = Iterates_NCRPG_bt[end]
            time_NCRPG_bt      = time(median(bm_NCRPG_bt))/1e9
            obj_NCRPG_bt       = f(OB, res_NCRPG_bt)
            spar_NCRPG_bt      = sum(abs.(res_NCRPG_bt).< 1e-8)/n/p
            it_NCRPG_bt        = length(Iterates_NCRPG_bt)
            orth_NCRPG_bt      = norm(res_NCRPG_bt'*res_NCRPG_bt - I(p))
            # RPG
            res_RPG         = Iterates_RPG[end]
            time_RPG        = time(median(bm_RPG))/1e9
            obj_RPG         = f(OB, res_RPG)
            spar_RPG        = sum(abs.(res_RPG).< 1e-8)/n/p
            orth_RPG        = norm(res_RPG'*res_RPG - I(p))
            #
            # Update results
            # Time values
            time_NCRPG_tmp[c]      += time_NCRPG
            time_NCRPG_bt_tmp[c]   += time_NCRPG_bt
            time_RPG_tmp[c]        += time_RPG
            # Objective values
            obj_NCRPG_tmp[c]       += obj_NCRPG
            obj_NCRPG_bt_tmp[c]    += obj_NCRPG_bt
            obj_RPG_tmp[c]         += obj_RPG
            # Sparsity values
            spar_NCRPG_tmp[c]      += spar_NCRPG
            spar_NCRPG_bt_tmp[c]   += spar_NCRPG_bt
            spar_RPG_tmp[c]        += spar_RPG
            # Orthogonality values
            orth_NCRPG_tmp[c]      += orth_NCRPG
            orth_NCRPG_bt_tmp[c]   += orth_NCRPG_bt
            orth_RPG_tmp[c]        += orth_RPG
            # Iteration values
            it_NCRPG_tmp[c]        += it_NCRPG
            it_NCRPG_bt_tmp[c]     += it_NCRPG_bt
            it_RPG_tmp[c]          += it_RPG
        end
    end
    for (c, μ) in enumerate(μs)
        push!(df_results_RPG,
            [μ, n, p, time_RPG_tmp[c]/m_tests, obj_RPG_tmp[c]/m_tests, spar_RPG_tmp[c]/m_tests, it_RPG_tmp[c]/m_tests, orth_RPG_tmp[c]/m_tests]
        )
        push!(df_results_NCRPG,
            [μ, n, p, time_NCRPG_tmp[c]/m_tests, obj_NCRPG_tmp[c]/m_tests, spar_NCRPG_tmp[c]/m_tests, it_NCRPG_tmp[c]/m_tests, orth_NCRPG_tmp[c]/m_tests]
        )
        push!(df_results_NCRPG_bt,
            [μ, n, p, time_NCRPG_bt_tmp[c]/m_tests, obj_NCRPG_bt_tmp[c]/m_tests, spar_NCRPG_bt_tmp[c]/m_tests, it_NCRPG_bt_tmp[c]/m_tests, orth_NCRPG_bt_tmp[c]/m_tests]
        )
    end
    #
    # Reset data collection variables
    time_RPG_tmp      .= zeros(length(μs))
    time_NCRPG_tmp    .= zeros(length(μs))
    time_NCRPG_bt_tmp .= zeros(length(μs))
    obj_RPG_tmp       .= zeros(length(μs))
    obj_NCRPG_tmp     .= zeros(length(μs))
    obj_NCRPG_bt_tmp  .= zeros(length(μs))
    spar_RPG_tmp      .= zeros(length(μs))
    spar_NCRPG_tmp    .= zeros(length(μs))
    spar_NCRPG_bt_tmp .= zeros(length(μs))
    it_RPG_tmp        .= zeros(length(μs))
    it_NCRPG_tmp      .= zeros(length(μs))
    it_NCRPG_bt_tmp   .= zeros(length(μs))
    orth_RPG_tmp      .= zeros(length(μs))
    orth_NCRPG_tmp    .= zeros(length(μs))
    orth_NCRPG_bt_tmp .= zeros(length(μs))
end
```

We export the results to CSV files

<details class="code-fold">
<summary>Code</summary>

``` julia
# Sort the dataframes by the parameter μ and create the final results dataframes
df_results_NCRPG = sort(df_results_NCRPG, :μ)
df_results_NCRPG_bt = sort(df_results_NCRPG_bt, :μ)
df_results_RPG = sort(df_results_RPG, :μ)
df_results_time_iter = DataFrame(
    μ             = df_results_NCRPG.μ,
    n             = Int.(df_results_NCRPG.n),
    p             = Int.(df_results_NCRPG.p),
    NCRPG_time     = df_results_NCRPG.time,
    NCRPG_iter     = Int.(round.(df_results_NCRPG.iterations, digits = 0)),
    NCRPG_bt_time  = df_results_NCRPG_bt.time,
    NCRPG_bt_iter  = Int.(round.(df_results_NCRPG_bt.iterations, digits = 0)),
    RPG_time     = df_results_RPG.time,
    RPG_iter     = Int.(round.(df_results_RPG.iterations, digits = 0)),
)
df_results_obj_spar_orth = DataFrame(
    μ               = df_results_NCRPG.μ,
    n               = Int.(df_results_NCRPG.n),
    p               = Int.(df_results_NCRPG.p),
    NCRPG_obj       = df_results_NCRPG.objective,
    NCRPG_sparsity  = df_results_NCRPG.sparsity,
    NCRPG_orth      = df_results_NCRPG.orthogonality,
    NCRPG_bt_obj    = df_results_NCRPG_bt.objective,
    NCRPG_bt_sparsity = df_results_NCRPG_bt.sparsity,
    NCRPG_bt_orth   = df_results_NCRPG_bt.orthogonality,
    RPG_obj         = df_results_RPG.objective,
    RPG_sparsity    = df_results_RPG.sparsity,
    RPG_orth        = df_results_RPG.orthogonality,
)
# Write the results to CSV files
CSV.write(joinpath(results_folder, "results-OB-time-iter-$(m_tests).csv"), df_results_time_iter)
CSV.write(joinpath(results_folder, "results-OB-obj-spar-orth-$(m_tests).csv"), df_results_obj_spar_orth)
```

</details>

We can take a look at how the algorithms compare to each other in their performance with the following tables.
First, we look at the time and number of iterations for each algorithm.

| **μ** | **n** | **p** | **NCRPG_const_time** | **NCRPG_const_iter** | **NCRPG_bt_time** | **NCRPG_bt_iter** | **RPG_time** | **RPG_iter** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100 | 5 | 0.748053 | 42874 | 0.578981 | 6234 | 1.25998 | 42874 |
| 0.1 | 200 | 5 | 1.29457 | 31795 | 0.711506 | 3494 | 2.02036 | 31814 |
| 0.1 | 300 | 5 | 3.19935 | 40532 | 1.5 | 3694 | 4.65004 | 40535 |
| 0.5 | 100 | 5 | 0.196853 | 10588 | 0.0955647 | 985 | 0.299515 | 10590 |
| 0.5 | 200 | 5 | 0.627545 | 14398 | 0.288494 | 1230 | 0.930948 | 14407 |
| 0.5 | 300 | 5 | 2.16516 | 26057 | 0.631828 | 1292 | 2.94764 | 26080 |
| 1.0 | 100 | 5 | 0.178784 | 9705 | 0.180771 | 1472 | 0.262947 | 9732 |
| 1.0 | 200 | 5 | 0.237726 | 5903 | 0.271266 | 626 | 0.35452 | 5911 |
| 1.0 | 300 | 5 | 0.0358025 | 449 | 0.00785541 | 27 | 0.0564122 | 449 |

Second, we look at the objective values, sparsity, and orthogonality of the solutions found by each algorithm.

| **μ** | **n** | **p** | **NCRPG_const_obj** | **NCRPG_const_spar** | **NCRPG_const_orth** | **NCRPG_bt_obj** | **NCRPG_bt_spar** | **NCRPG_bt_orth** | **RPG_obj** | **RPG_spar** | **RPG_orth** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100 | 5 | 3.22058 | 0.4718 | 0.15632 | 3.22006 | 0.4752 | 0.155214 | 3.22058 | 0.4718 | 0.156319 |
| 0.1 | 200 | 5 | 4.39565 | 0.5177 | 0.116374 | 4.39581 | 0.5171 | 0.118228 | 4.39565 | 0.5177 | 0.116374 |
| 0.1 | 300 | 5 | 5.24113 | 0.552267 | 0.101485 | 5.242 | 0.5516 | 0.100302 | 5.24113 | 0.552267 | 0.101485 |
| 0.5 | 100 | 5 | 13.0595 | 0.7348 | 0.123164 | 13.0992 | 0.7344 | 0.117375 | 13.0595 | 0.7348 | 0.123164 |
| 0.5 | 200 | 5 | 16.8825 | 0.813 | 0.0732099 | 16.8633 | 0.8117 | 0.0775427 | 16.8825 | 0.813 | 0.0732099 |
| 0.5 | 300 | 5 | 19.159 | 0.872133 | 0.0559839 | 19.1961 | 0.873333 | 0.0595898 | 19.159 | 0.872133 | 0.0559839 |
| 1.0 | 100 | 5 | 22.1209 | 0.8722 | 0.0602495 | 22.0776 | 0.8728 | 0.0722475 | 22.1209 | 0.8722 | 0.0602495 |
| 1.0 | 200 | 5 | 25.5964 | 0.9794 | 2.33716e-16 | 25.6114 | 0.9824 | 0.0429794 | 25.5964 | 0.9794 | 2.43844e-16 |
| 1.0 | 300 | 5 | 24.7444 | 0.996667 | 0.0 | 24.7457 | 0.996667 | 0.0 | 24.7444 | 0.996667 | 0.0 |

## Technical details

This tutorial is cached. It was last run on the following package versions.

<details class="code-fold">
<summary>Code</summary>

``` julia
using Pkg
Pkg.status()
```

</details>

    Status `~/Repositories/Julia/ManoptExamples.jl/examples/Project.toml`
      [6e4b80f9] BenchmarkTools v1.6.0
      [336ed68f] CSV v0.10.15
      [13f3f980] CairoMakie v0.15.6
      [0ca39b1e] Chairmarks v1.3.1
      [35d6a980] ColorSchemes v3.31.0
      [5ae59095] Colors v0.13.1
      [a93c6f00] DataFrames v1.8.0
      [31c24e10] Distributions v0.25.122
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
      [08abe8d2] PrettyTables v3.1.0
      [6099a3de] PythonCall v0.9.28
      [f468eda6] QuadraticModels v0.9.14
      [1e40b3f8] RipQP v0.7.0
    Info Packages marked with ⌅ have new versions available but compatibility constraints restrict them from upgrading. To see why use `status --outdated`

This tutorial was last rendered October 15, 2025, 19:20:36.

## Literature

```@bibliography
Pages = ["NCRPG-Sparse-PCA.md"]
Canonical=false
```
