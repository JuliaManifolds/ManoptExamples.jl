# Sparse PCA

Paula John, Hajg Jasa
2025-10-01

## Introduction

In this example we use the Nonconvex Riemannian Proximal Gradient (NCRPG) method [BergmannJasaJohnPfeffer:2025:1](@cite) and compare it to the Riemannian Proximal Gradient (RPG) method [HuangWei:2021:1](@cite) and to the Manifold Proximal Gradient method (ManPG) [ChenMaMan-ChoSoZhan:2020](@cite).
This example reproduces the results from [BergmannJasaJohnPfeffer:2025:1](@cite), Section 6.1.
The numbers may vary slightly.

``` julia
using PrettyTables
using BenchmarkTools
using CSV, DataFrames, Dates
using ColorSchemes, Plots, LaTeXStrings
using Random, LinearAlgebra, LRUCache, Distributions
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
```

## The Problem

Let $\mathcal M = \mathrm{OB}(n,r)$ be the oblique manifold, i.e., the set of $n \times r$ matrices with unit-norm columns.
Let $g \colon \mathcal M \to \mathbb R$ be defined by

``` math
g(X) = \frac{1}{2} \Vert X^\top A^\top A X - D^2 \Vert^2,
```

where $A \in \mathbb R^{m \times n}$ is a data matrix, $D = \mathrm{diag}(d_1, \ldots, d_r)$ is a diagonal matrix containing the top $r$ singular values of $A$, and $\Vert \cdot \Vert$ is the Frobenius norm.

Let $h \colon \mathcal M \to \mathbb R$ be defined by

``` math
h(X) = \mu \Vert X \Vert_1
```

be the sparsity-enforcing term given by the $\ell_1$-norm, where $\mu \ge 0$ is a regularization parameter.

We define our total objective function as $f = g + h$.
The goal is to find the minimizer of $f$ on $\mathcal M$, which is heuristically the point that diagonalizes $A^\top A$ as much as possible while being sparse.

## Numerical Experiment

We initialize the experiment parameters, as well as some utility functions.

``` julia
# Set random seed for reproducibility
random_seed = 1520
Random.seed!(random_seed)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2.0
m_tests = 10 # number of tests for each parameter setting
means = 20 # number of means to compute

atol = 1e-4
max_iters = 100000
n_p_array = [(100, 5), (200, 5), (300, 5)]
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

We also introduce an implementation of the ManPG algorithm for the Sparse PCA problem on the oblique manifold, following [ChenMaMan-ChoSoZhan:2020](@cite).

``` julia
function prox_l1norm(B,λ)
    A = abs.(B) .- λ
    Act_set = A .> 0  
    x_prox = Act_set .* sign.(B) .* A
    Inact_set = (A .<= 0)
    return x_prox, Act_set, Inact_set
end
#
function Semi_newton_Sphere(n, X, t, B, μt, inner_tol, prox_fun, inner_max_iter, Lam0)
    Lam = Lam0
    X_Lam_prod = B+ t * Lam * X 
    Z, Act_set, Inact_set = prox_fun(X_Lam_prod, μt)
    ZX = Z'X
    R_Lam = ZX - 1.0
    RE = R_Lam
    r_l = abs(R_Lam)
    
    λ = 0.2
    j = 0
    
    while r_l > inner_tol
        reg = λ * max(min(r_l, 0.1), 1e-11)
        G = 4 * t * X' * (Act_set.*X)
        new_d = - RE/(G + reg)
        t_new = 1.0
        X_d_prod = t * new_d * X 
        X_Lam_new_prod = X_Lam_prod + t_new * X_d_prod
        Z, Act_set, Inact_set = prox_fun(X_Lam_new_prod,μt)
        ZX = Z'X
        R_Lam_new =  ZX -1.0
        
        r_l_new = abs(R_Lam_new)

        #backtracking of new direction 
        while r_l_new^2 >= r_l^2 * (1 - 0.001 * t_new) && t_new > 1e-3
            t_new *= 0.5
            X_Lam_new_prod = X_Lam_prod + t_new * X_d_prod
            Z, Act_set, Inact_set = prox_fun(X_Lam_new_prod,μt)
            ZX = Z'X
            R_Lam_new = ZX -1.0
            r_l_new = abs(R_Lam_new)
        end
        Lam += t_new * new_d
        r_l = r_l_new
        R_Lam = R_Lam_new
        RE = R_Lam
        X_Lam_prod = X_Lam_new_prod

        if j > inner_max_iter
            break
        end
        j += 1
    end
    return Z, Lam
end
#
function manpg_SPCA_OB(
    M,
    H, # AtA
    D,
    μ, #sparsity param
    step_size,
    start; 
    max_iters    = Int(1e3), 
    tol       = 1e-8, 
    inner_it  = 10,
    inner_tol = 1e-12,
    bt_param  = 0.5,
    α_min     = 1e-8, #minimum stepsize factor 
    record    = false
)
    # Parameters
    n, p = size(start)

    cost_fun(M,X, HX) =  0.5 * norm(X'HX - D)^2 + μ * norm(X, 1)
    h(X) = μ * norm(X, 1)
    prox_fun(b, λ) = prox_l1norm(b, λ)
    X = start 
    HX = H * X 
    F_old = cost_fun(M, X, HX)

    Lam = zeros(p)
    Desc = Array{Float64}(undef,n,p)
    it_end = max_iters

    if !record
        for it in 1:max_iters
            ngX = -2 * HX * (X'HX - D) # negative Euclidean gradient 
            for r in 1:p # Semi Newton for each component
                PY, Lam[r] = Semi_newton_Sphere(
                    n, 
                    X[:, r], 
                    step_size, 
                    X[:, r] + step_size * ngX[:, r],
                    μ * step_size, 
                    inner_tol, 
                    prox_fun, 
                    inner_it, 
                    Lam[r]
                )
                Desc[:, r] = PY - X[:, r]
            end
            α = 1.0
            Z = retract(M, X, α * Desc)
            HZ = H * Z
            F_trial = cost_fun(M, Z, HZ)
            normDesc = norm(Desc, 2)
            normDescsquared = normDesc^2
            
            # Linesearch
            while F_trial >= F_old - 0.5 / step_size * α * normDescsquared
                α *= bt_param
                if α < α_min
                    break
                end

                Z = retract(M, X, α * Desc)
                HZ = H * Z 
                F_trial = cost_fun(M, Z, HZ)
            end

            X = Z
            HX = HZ
            F_old = F_trial
            
            if normDesc/step_size * α < tol
                it_end = it
                break 
            end

        end
        X_manpg = X
        return X_manpg, it_end
    #record iterates
    else 
        Iterates = []
        for it in 1:max_iters
            ngX = -2 * HX * (X'HX - D) # negative Euclidean gradient 
            for r in 1:p #Semi newton for each component
                PY, Lam[r] = Semi_newton_Sphere(
                    n, 
                    X[:, r], 
                    step_size, 
                    X[:, r] + step_size * ngX[:, r],
                    μ * step_size, 
                    inner_tol, 
                    prox_fun, 
                    inner_it, 
                    Lam[r]
                )
                Desc[:, r] = PY - X[:, r]
            end
            α = 1.0
            Z = retract(M, X, α * Desc)
            HZ = H * Z
            F_trial = cost_fun(M, Z, HZ)
            normDesc = norm(Desc, 2)
            normDescsquared = normDesc^2
            
            # Linesearch
            while F_trial >= F_old - 0.5 / step_size * α * normDescsquared
                α *= bt_param
                if α < α_min
                    break
                end

                Z = retract(M, X, α * Desc)
                HZ = H * Z 
                F_trial = cost_fun(M, Z, HZ)
            end

            X = Z
            HX = HZ
            F_old = F_trial
            push!(Iterates, copy(X))
            if normDesc/step_size < tol
                it_end = it
                break 
            end

        end
        return Iterates, it_end
    end
end
```

    manpg_SPCA_OB (generic function with 1 method)

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
                record = [:Iteration, :Iterate, :Time],
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
            # ManPG
            Iterates_ManPG, it_ManPG = manpg_SPCA_OB(OB, H, D, μ, step_size, start;
                max_iters = max_iters, 
                tol = stop_NCRPG,
                record = true
            )
            bm_ManPG = @benchmark manpg_SPCA_OB($OB, $H, $D, $μ, $step_size, $start;
                max_iters = $max_iters, 
                tol = $stop_NCRPG
            )
            # 
            # Collect test results
            Iterates_NCRPG  = get_record(rec_NCRPG, :Iteration, :Iterate)
            res_NCRPG       = Iterates_NCRPG[end]
            time_NCRPG      = time(median(bm_NCRPG)) / 1e9
            obj_NCRPG       = f(OB, res_NCRPG)
            spar_NCRPG      = sum(abs.(res_NCRPG) .< 1e-8) / n / p
            it_NCRPG        = length(Iterates_NCRPG)
            orth_NCRPG      = norm(res_NCRPG' * res_NCRPG - I(p))
            # NCRPG with backtracking
            Iterates_NCRPG_bt  = get_record(rec_NCRPG_bt, :Iteration, :Iterate)
            res_NCRPG_bt       = Iterates_NCRPG_bt[end]
            time_NCRPG_bt      = time(median(bm_NCRPG_bt)) / 1e9
            obj_NCRPG_bt       = f(OB, res_NCRPG_bt)
            spar_NCRPG_bt      = sum(abs.(res_NCRPG_bt) .< 1e-8) / n / p
            it_NCRPG_bt        = length(Iterates_NCRPG_bt)
            orth_NCRPG_bt      = norm(res_NCRPG_bt' * res_NCRPG_bt - I(p))
            # RPG
            res_RPG         = Iterates_RPG[end]
            time_RPG        = time(median(bm_RPG)) / 1e9
            obj_RPG         = f(OB, res_RPG)
            spar_RPG        = sum(abs.(res_RPG) .< 1e-8) / n / p
            orth_RPG        = norm(res_RPG' * res_RPG - I(p))
            # ManPG
            res_ManPG         = Iterates_ManPG[end]
            time_ManPG        = time(median(bm_ManPG)) / 1e9
            obj_ManPG         = f(OB, res_ManPG)
            spar_ManPG        = sum(abs.(res_ManPG) .< 1e-8) / n / p
            orth_ManPG        = norm(res_ManPG' * res_ManPG - I(p))

            # Save function value errors and gradient mapping norm for one case
            if c == 2 && n == 100 # i.e. μ = 0.5
                ###### Optimal Solution ######
                rec_NCRPG_opt = proximal_gradient_method(OB, f, g, grad_g, start; 
                    prox_nonsmooth = prox_norm1_NCRPG, 
                    stepsize = ConstantLength(step_size/2),
                    record = [:Iteration, :Iterate],
                    return_state = true,
                    stopping_criterion = StopAfterIteration(10*max_iters)| StopWhenGradientMappingNormLess(1e-8)
                )
                Iterates_OPT = get_record(rec_NCRPG_opt, :Iteration, :Iterate)
                fs_opt = f.(Ref(OB), Iterates_OPT)
              
                ############## Cost function ################
                fs_RPG      = f.(Ref(OB), Iterates_RPG)
                fs_ManPG    = f.(Ref(OB), Iterates_ManPG)
                fs_NCRPG    = f.(Ref(OB), Iterates_NCRPG)
                fs_NCRPG_bt = f.(Ref(OB), Iterates_NCRPG_bt)

                f_opt_RPG       = f(OB, res_RPG)
                f_opt_ManPG     = f(OB, res_ManPG)
                f_opt_NCRPG     = f(OB, res_NCRPG)
                f_opt_NCRPG_bt  = f(OB, res_NCRPG_bt)
                f_opt           = minimum(fs_opt)

                diff_fs_RPG      = fs_RPG .- f_opt
                diff_fs_ManPG    = fs_ManPG .- f_opt
                diff_fs_NCRPG    = fs_NCRPG .- f_opt
                diff_fs_NCRPG_bt = fs_NCRPG_bt .- f_opt

                minimum(diff_fs_NCRPG)
                minimum(diff_fs_NCRPG_bt)
                minimum(diff_fs_RPG)
                minimum(diff_fs_ManPG)

                ############### Time ################
                Times_RPG       = range(0.0, time_RPG,   length=length(Iterates_RPG))
                Times_ManPG     = range(0.0, time_ManPG, length=length(Iterates_ManPG))
                Times_NCRPG     = range(0.0, time_NCRPG, length=length(Iterates_NCRPG))

                Times_NCRPG_bt = Dates.value.(get_record(rec_NCRPG_bt, :Iteration, :Time))/1e9
                Times_NCRPG_bt = Times_NCRPG_bt .- Times_NCRPG_bt[1]
                Times_NCRPG_bt = Times_NCRPG_bt./Times_NCRPG_bt[end]*time_NCRPG_bt

                ######### Data Frames function value ############
                indices_NCRPG    = vcat(1:40:it_NCRPG-1, it_NCRPG)
                indices_RPG      = vcat(1:40:it_RPG-1, it_RPG)
                indices_ManPG    = vcat(1:40:it_ManPG-1, it_ManPG)
                indices_NCRPG_bt = vcat(1:3:it_NCRPG_bt-1, it_NCRPG_bt)

                df_diff_fs_NCRPG    = DataFrame(time = Times_NCRPG[indices_NCRPG], diff_fs = diff_fs_NCRPG[indices_NCRPG])
                df_diff_fs_NCRPG_bt = DataFrame(time = Times_NCRPG_bt[indices_NCRPG_bt], diff_fs = diff_fs_NCRPG_bt[indices_NCRPG_bt])
                df_diff_fs_RPG      = DataFrame(time = Times_RPG[indices_RPG], diff_fs = diff_fs_RPG[indices_RPG])
                df_diff_fs_ManPG    = DataFrame(time = Times_ManPG[indices_ManPG], diff_fs = diff_fs_ManPG[indices_ManPG])

                ######### Data Frames gradient mapping ############
                indices_NCRPG_2    = vcat(2:40:it_NCRPG-1, it_NCRPG)
                indices_RPG_2      = vcat(2:40:it_RPG-1, it_RPG)
                indices_ManPG_2    = vcat(2:40:it_ManPG-1, it_ManPG)
                indices_NCRPG_bt_2 = vcat(2:3:it_NCRPG_bt-2, it_NCRPG_bt)

                grad_maps_NCRPG = 1/step_size * [distance(OB, Iterates_NCRPG[i], Iterates_NCRPG[i-1]) for i in indices_NCRPG_2]
                grad_maps_RPG   = L * [distance(OB, Iterates_RPG[i],   Iterates_RPG[i-1])   for i in indices_RPG_2  ]
                grad_maps_ManPG = L * [distance(OB, Iterates_ManPG[i], Iterates_ManPG[i-1]) for i in indices_ManPG_2]
                # Record gradient mapping for backtracking 
                grad_maps_NCRPG_bt = []
                for i in indices_NCRPG_bt_2
                    rec_NCRPG_bt_i = proximal_gradient_method(OB, f, g, grad_g, start; 
                        prox_nonsmooth = prox_norm1_NCRPG,
                        stepsize = ProximalGradientMethodBacktracking(; 
                            strategy=:nonconvex, 
                            initial_stepsize=init_step_size_bt,
                            stop_when_stepsize_less = 1e-7
                        ),
                        return_state = true,
                        stopping_criterion =    StopAfterIteration(i)| StopWhenGradientMappingNormLess(stop_NCRPG_bt)
                    )
                    step_size_i = rec_NCRPG_bt_i.last_stepsize
                    push!(grad_maps_NCRPG_bt, 1/step_size_i*distance(OB, Iterates_NCRPG_bt[i], Iterates_NCRPG_bt[i-1]))
                end

                df_grad_maps_NCRPG    = DataFrame(time = Times_NCRPG[indices_NCRPG_2], grad_maps = grad_maps_NCRPG)
                df_grad_maps_NCRPG_bt = DataFrame(time = Times_NCRPG_bt[indices_NCRPG_bt_2], grad_maps = grad_maps_NCRPG_bt)
                df_grad_maps_RPG      = DataFrame(time = Times_RPG[indices_RPG_2],     grad_maps = grad_maps_RPG)
                df_grad_maps_ManPG    = DataFrame(time = Times_ManPG[indices_ManPG_2], grad_maps = grad_maps_ManPG)
                
                # Write data
                CSV.write(joinpath(results_folder, "diff_fs_NCRPG.csv"), df_diff_fs_NCRPG; delim=" ", header=false)
                CSV.write(joinpath(results_folder, "diff_fs_NCRPG_bt.csv"), df_diff_fs_NCRPG_bt; delim=" ", header=false)
                CSV.write(joinpath(results_folder, "diff_fs_RPG.csv"), df_diff_fs_RPG; delim=" ", header=false)
                CSV.write(joinpath(results_folder, "diff_fs_ManPG.csv"), df_diff_fs_ManPG; delim=" ", header=false)
                CSV.write(joinpath(results_folder, "grad_maps_NCRPG.csv"), df_grad_maps_NCRPG; delim=" ", header=false)
                CSV.write(joinpath(results_folder, "grad_maps_NCRPG_bt.csv"), df_grad_maps_NCRPG_bt; delim=" ", header=false)
                CSV.write(joinpath(results_folder, "grad_maps_RPG.csv"), df_grad_maps_RPG; delim=" ", header=false)
                CSV.write(joinpath(results_folder, "grad_maps_ManPG.csv"), df_grad_maps_ManPG; delim=" ", header=false)
            end
            #
            # Update results
            # Time values
            time_NCRPG_tmp[c]      += time_NCRPG
            time_NCRPG_bt_tmp[c]   += time_NCRPG_bt
            time_RPG_tmp[c]        += time_RPG
            time_ManPG_tmp[c]      += time_ManPG
            # Objective values
            obj_NCRPG_tmp[c]       += obj_NCRPG
            obj_NCRPG_bt_tmp[c]    += obj_NCRPG_bt
            obj_RPG_tmp[c]         += obj_RPG
            obj_ManPG_tmp[c]       += obj_ManPG
            # Sparsity values
            spar_NCRPG_tmp[c]      += spar_NCRPG
            spar_NCRPG_bt_tmp[c]   += spar_NCRPG_bt
            spar_RPG_tmp[c]        += spar_RPG
            spar_ManPG_tmp[c]      += spar_ManPG
            # Orthogonality values
            orth_NCRPG_tmp[c]      += orth_NCRPG
            orth_NCRPG_bt_tmp[c]   += orth_NCRPG_bt
            orth_RPG_tmp[c]        += orth_RPG
            orth_ManPG_tmp[c]      += orth_ManPG
            # Iteration values
            it_NCRPG_tmp[c]        += it_NCRPG
            it_NCRPG_bt_tmp[c]     += it_NCRPG_bt
            it_RPG_tmp[c]          += it_RPG
            it_ManPG_tmp[c]        += it_ManPG
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
        push!(df_results_ManPG, 
            [μ, n, p, time_ManPG_tmp[c]/m_tests, obj_ManPG_tmp[c]/m_tests, spar_ManPG_tmp[c]/m_tests, it_ManPG_tmp[c]/m_tests, orth_ManPG_tmp[c]/m_tests]
        )
    end
    #
    # Reset data collection variables
    time_RPG_tmp      .= zeros(length(μs))
    time_NCRPG_tmp    .= zeros(length(μs))
    time_NCRPG_bt_tmp .= zeros(length(μs))
    time_ManPG_tmp    .= zeros(length(μs))
    obj_RPG_tmp       .= zeros(length(μs))
    obj_NCRPG_tmp     .= zeros(length(μs))
    obj_NCRPG_bt_tmp  .= zeros(length(μs))
    obj_ManPG_tmp     .= zeros(length(μs))
    spar_RPG_tmp      .= zeros(length(μs))
    spar_NCRPG_tmp    .= zeros(length(μs))
    spar_NCRPG_bt_tmp .= zeros(length(μs))
    spar_ManPG_tmp    .= zeros(length(μs))
    it_RPG_tmp        .= zeros(length(μs))
    it_NCRPG_tmp      .= zeros(length(μs))
    it_NCRPG_bt_tmp   .= zeros(length(μs))
    it_ManPG_tmp      .= zeros(length(μs))
    orth_RPG_tmp      .= zeros(length(μs))
    orth_NCRPG_tmp    .= zeros(length(μs))
    orth_NCRPG_bt_tmp .= zeros(length(μs))
    orth_ManPG_tmp    .= zeros(length(μs))
end
```

We export the results to CSV files

``` julia
# Sort the dataframes by the parameter μ and create the final results dataframes
df_results_NCRPG = sort(df_results_NCRPG, :μ)
df_results_NCRPG_bt = sort(df_results_NCRPG_bt, :μ)
df_results_RPG = sort(df_results_RPG, :μ)
df_results_ManPG = sort(df_results_ManPG, :μ)
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
    ManPG_time     = df_results_ManPG.time,
    ManPG_iter     = Int.(round.(df_results_ManPG.iterations, digits = 0)),
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
    ManPG_obj       = df_results_ManPG.objective,
    ManPG_sparsity  = df_results_ManPG.sparsity,
    ManPG_orth      = df_results_ManPG.orthogonality,
)
# Write the results to CSV files
CSV.write(joinpath(results_folder, "results-OB-time-iter-$(m_tests).csv"), df_results_time_iter)
CSV.write(joinpath(results_folder, "results-OB-obj-spar-orth-$(m_tests).csv"), df_results_obj_spar_orth)
```

We can take a look at how the algorithms compare to each other in their performance with the following tables.
First, we look at the time and number of iterations for each algorithm.

| **μ** | **n** | **p** | **NCRPGconst_time** | **NCRPGconst_iter** | **NCRPGbt_time** | **NCRPGbt_iter** | **RPG_time** | **RPG_iter** | **ManPG_time** | **ManPG_iter** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100 | 5 | 0.217693 | 12382 | 0.170006 | 2009 | 0.334099 | 12381 | 0.823055 | 12427 |
| 0.1 | 200 | 5 | 0.588867 | 14645 | 0.273925 | 1572 | 0.864565 | 14644 | 2.03126 | 22305 |
| 0.1 | 300 | 5 | 1.43485 | 18408 | 0.620402 | 2066 | 2.03658 | 18410 | 4.09954 | 18619 |
| 0.5 | 100 | 5 | 0.10791 | 5675 | 0.0311343 | 409 | 0.158505 | 5781 | 0.35857 | 14308 |
| 0.5 | 200 | 5 | 0.356628 | 8249 | 0.10045 | 596 | 0.506606 | 8251 | 1.19806 | 8500 |
| 0.5 | 300 | 5 | 1.00542 | 12234 | 0.292168 | 913 | 1.36578 | 12258 | 2.79998 | 12607 |
| 1.0 | 100 | 5 | 0.0717333 | 3750 | 0.0394211 | 423 | 0.103123 | 3754 | 0.243326 | 3877 |
| 1.0 | 200 | 5 | 0.137224 | 3384 | 0.0475742 | 239 | 0.20263 | 3389 | 0.439992 | 3491 |
| 1.0 | 300 | 5 | 0.0621101 | 792 | 0.00885604 | 31 | 0.0915075 | 792 | 0.17637 | 30691 |

Second, we look at the objective values, sparsity, and orthogonality of the solutions found by each algorithm.

| **μ** | **n** | **p** | **NCRPGconst_obj** | **NCRPGconst_spar** | **NCRPGconst_orth** | **NCRPGbt_obj** | **NCRPGbt_spar** | **NCRPGbt_orth** | **RPG_obj** | **RPG_spar** | **RPG_orth** | **ManPG_obj** | **ManPG_spar** | **ManPG_orth** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.1 | 100 | 5 | 3.21299 | 0.4746 | 0.139607 | 3.21298 | 0.4756 | 0.139187 | 3.21299 | 0.4746 | 0.139607 | 3.21299 | 0.4746 | 0.139607 |
| 0.1 | 200 | 5 | 4.36851 | 0.5204 | 0.124799 | 4.36942 | 0.518 | 0.125262 | 4.36851 | 0.5204 | 0.124799 | 4.36851 | 0.5204 | 0.1248 |
| 0.1 | 300 | 5 | 5.19944 | 0.557867 | 0.0983619 | 5.19917 | 0.557733 | 0.0974957 | 5.19944 | 0.557867 | 0.0983619 | 5.19944 | 0.557867 | 0.098362 |
| 0.5 | 100 | 5 | 13.0335 | 0.7366 | 0.117771 | 13.0347 | 0.7356 | 0.12262 | 13.0335 | 0.7366 | 0.117771 | 13.0335 | 0.7366 | 0.117771 |
| 0.5 | 200 | 5 | 16.8819 | 0.8121 | 0.0912683 | 16.8785 | 0.8155 | 0.0890128 | 16.8819 | 0.8121 | 0.0912683 | 16.8819 | 0.8121 | 0.0912676 |
| 0.5 | 300 | 5 | 19.0018 | 0.874333 | 0.0570498 | 19.0483 | 0.8756 | 0.0604583 | 19.0018 | 0.874333 | 0.0570493 | 19.0018 | 0.874333 | 0.0570505 |
| 1.0 | 100 | 5 | 21.9124 | 0.8762 | 0.0605361 | 21.9033 | 0.8714 | 0.054258 | 21.9124 | 0.8762 | 0.0605361 | 21.9124 | 0.8762 | 0.0605361 |
| 1.0 | 200 | 5 | 25.4645 | 0.9783 | 0.0182046 | 25.5534 | 0.984 | 1.22125e-16 | 25.4645 | 0.9783 | 0.0182045 | 25.4645 | 0.9783 | 0.0182037 |
| 1.0 | 300 | 5 | 24.4549 | 0.9966 | 1.11022e-17 | 24.4503 | 0.996667 | 0.0 | 24.4549 | 0.9966 | 1.11022e-17 | 24.4549 | 0.9966 | 1.00251e-6 |

## Technical details

This tutorial is cached. It was last run on the following package versions.

``` julia
using Pkg
Pkg.status()
```

    Status `~/Repositories/Julia/ManoptExamples.jl/examples/Project.toml`
      [6e4b80f9] BenchmarkTools v1.8.0
      [336ed68f] CSV v0.10.16
    ⌃ [13f3f980] CairoMakie v0.15.11
      [0ca39b1e] Chairmarks v1.3.1
      [35d6a980] ColorSchemes v3.31.0
      [5ae59095] Colors v0.13.1
      [a93c6f00] DataFrames v1.8.2
    ⌃ [31c24e10] Distributions v0.25.126
    ⌃ [e9467ef8] GLMakie v0.13.11
      [4d00f742] GeometryTypes v0.8.5
      [7073ff75] IJulia v1.34.4
      [682c06a0] JSON v1.6.1
      [8ac3fa9e] LRUCache v1.6.2
      [b964fa9f] LaTeXStrings v1.4.0
      [d3d80556] LineSearches v7.7.1
    ⌅ [ee78f7c6] Makie v0.24.11
      [af67fdf4] ManifoldDiff v0.4.5
    ⌃ [1cead3c2] Manifolds v0.11.27
    ⌃ [3362f125] ManifoldsBase v2.4.0
    ⌃ [0fc0a36d] Manopt v0.5.39
      [5b8d5e80] ManoptExamples v0.1.18 `..`
      [51fcb6bd] NamedColors v0.2.3
      [6fe1bfb0] OffsetArrays v1.17.0
      [91a5bcdd] Plots v1.41.6
    ⌃ [08abe8d2] PrettyTables v3.3.2
      [6099a3de] PythonCall v0.9.35
      [f468eda6] QuadraticModels v0.9.16
    ⌃ [731186ca] RecursiveArrayTools v4.3.1
      [1e40b3f8] RipQP v0.7.0
    Info Packages marked with ⌃ and ⌅ have new versions available. Those with ⌃ may be upgradable, but those with ⌅ are restricted by compatibility constraints from upgrading. To see why use `status --outdated`

This tutorial was last rendered July 13, 2026, 21:17:38.

## Literature

```@bibliography
Pages = ["NCRPG-Sparse-PCA.md"]
Canonical=false
```
