#
# Findin the Riemannian median of a dataset on H^n and P(n)
# comparing the convex bundle method to
# 1. the proximal bundle method
# 2. the subgradient method
# 3. CPPA
#
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots#; pythonplot()
using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
#
# Settings
experiment_name = "Riemannian_Median"
results_folder = joinpath(@__DIR__, experiment_name)
export_orig = true
export_result = true
export_table = true
toggle_debug = false
!isdir(results_folder) && mkdir(results_folder)
#
# Parameters
Random.seed!(42)
atol = 1e-8# √eps()
# δ = -1.0
# μ = 0.5
#
# Colors
data_color = RGBA{Float64}(colorant"#BBBBBB")
noise_color = RGBA{Float64}(colorant"#33BBEE") # Tol Vibrant Teal
result_color = RGBA{Float64}(colorant"#EE7733") # Tol Vibrant Orange
#
# Utils
function close_point(M, p, tol)
    X = rand(M; vector_at=p)
    X .= tol * rand() * X / norm(M, p, X)
    return retract(M, p, X, Manifolds.default_retraction_method(M, typeof(p)))
end
#
# Riemannian median
function riemannian_median(M, n)
    if split(string(M), "(")[1] == "Hyperbolic"
        k_min = k_max = -1.0
    elseif split(string(M), "(")[1] == "Sphere"
        k_min = k_max = 1.0
    else
        k_min = nothing
        k_max = 0.0
    end
    #
    # Data
    if split(string(M), "(")[1] == "Sphere"
        north = [0.0 for _ in 1:manifold_dimension(M)]
        push!(north, 1.0)
        diameter = π / 4 #2 * π/7
        data = [close_point(M, north, diameter / 2) for _ in 1:n]
        p0 = data[1]
    else
        diameter = floatmax(Float64)
        data = [rand(M) for _ in 1:n]#[close_point(M, rand(M), diameter/2) for _ in 1:n]#
        dists = [distance(M, z, y) for z in data, y in data]
        p0 = data[minimum(Tuple(findmax(dists)[2]))] # data[findfirst(x -> x == diameter, dists)[1]]
    end
    #
    # Objective, subgradient and prox
    f(M, p) = sum(1 / length(data) * distance.(Ref(M), Ref(p), data))
    function ∂f(M, p)
        return sum(
            1 / length(data) *
            ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=atol),
        )
    end
    domf(M, p) = distance(M, p, p0) < diameter / 2 ? true : false
    # proxes = Function[
    #     (M, λ, p) -> prox_distance(M, λ / length(data), di, p, 1) for di in data
    # ]
    #
    # Optimization
    b = convex_bundle_method(
        M,
        f,
        ∂f,
        p0;
        diameter=diameter,
        domain=domf,
        k_min=k_min,
        k_max=k_max,
        count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
        debug=[
            :Iteration,
            (:Cost, "F(p): %1.16f "),
            (:ξ, "ξ: %1.16f "),
            # (:ϱ, "ϱ: %1.4f "),
            :Stop,
            1000,
            "\n",
        ],
        record=[:Iteration, :Cost, :Iterate],
        return_state=true,
        return_options=true,
        return_objective=true,
    )
    b_result = get_solver_result(b)
    b_record = get_record(b)
    #
    p = proximal_bundle_method(
        M,
        f,
        ∂f,
        p0;
        # δ=δ,
        # μ=μ,
        count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
        # stopping_criterion=StopWhenCostLess(f(M,b_result))|StopAfterIteration(10000),
        debug=[
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
        record=[:Iteration, :Cost, :Iterate],
        return_state=true,
    )
    p_result = get_solver_result(p)
    p_record = get_record(p)
    #
    s = subgradient_method(
        M,
        f,
        ∂f,
        p0;
        count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
        stepsize=DecreasingStepsize(1, 1, 0, 1, 0, :absolute),
        stopping_criterion=StopWhenSubgradientNormLess(1e-4) | StopAfterIteration(5000),
        debug=[:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
        record=[:Iteration, :Cost, :Iterate],
        return_state=true,
        return_options=true,
    )
    s_result = get_solver_result(s)
    s_record = get_record(s)
    #
    # c = cyclic_proximal_point(
    #     M,
    #     f,
    #     proxes,
    #     p0;
    #     # count=[:Cost, :ProximalMap],
    #     # cache=(:LRU, [:Cost, :ProximalMap], 50),
    #     # stopping_criterion=StopWhenCostLess(f(M,b_result))|StopWhenAny(StopAfterIteration(5000),StopWhenChangeLess(10.0^-8)),
    #     debug=[
    #         :Iteration,
    #         " | ",
    #         DebugProximalParameter(),
    #         " | ",
    #         (:Cost, "F(p): %1.16f "),
    #         " | ",
    #         :Change,
    #         "\n",
    #         1000,
    #         :Stop,
    #     ],
    #     record=[:Iteration, :Cost, :Iterate],
    #     return_state=true,
    #     return_options=true,
    # )
    # c_result = get_solver_result(c)
    # c_record = get_record(c)
    #
    # Benchmarking
    b_bm = @benchmark convex_bundle_method(
        $M,
        $f,
        $∂f,
        $p0;
        # bundle_size=$bundle_size,
        diameter=$diameter,
        domain=$domf,
        k_min=$k_min,
        k_max=$k_max,
        # count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
    )
    p_bm = @benchmark proximal_bundle_method(
        $M,
        $f,
        $∂f,
        $p0;
        # δ=$δ,
        # μ=$μ,
        # count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
    )
    s_bm = @benchmark subgradient_method(
        $M,
        $f,
        $∂f,
        $p0;
        # count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
        stepsize=DecreasingStepsize(1, 1, 0, 1, 0, :absolute),
        stopping_criterion=StopWhenSubgradientNormLess(1e-4) | StopAfterIteration(5000),
    )
    # c_bm = @benchmark cyclic_proximal_point(
    #     $M,
    #     $f,
    #     $proxes,
    #     $p0;
    #     # count=[:Cost, :ProximalMap],
    #     # cache=(:LRU, [:Cost, :ProximalMap], 50),
    #     # stopping_criterion=StopWhenCostLess($f($M,$b_result))|StopWhenAny(StopAfterIteration(5000),StopWhenChangeLess(10.0^-8))
    # )
    #
    # Results
    records = [
        b_record,
        p_record,
        s_record,
        # c_record
    ]
    times = [
        median(b_bm).time * 1e-9,
        median(p_bm).time * 1e-9,
        median(s_bm).time * 1e-9,
        # # median(c_bm).time * 1e-9,
    ]
    return records, times
end
#
# Finalize - export costs
for subexperiment_name in ["Sn"]#["SPD", "Hn", "Sn"]
    println(subexperiment_name)
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
        # e="Iterations",
        # f="Time (s)",
        # g="Objective",
    )
    CSV.write(
        joinpath(
            results_folder,
            experiment_name * "_$subexperiment_name" * "-Comparisons-Subgrad.csv",#"-Comparisons-Subgrad-CPPA.csv",
        ),
        A2;
        header=false,
    )
    if subexperiment_name == "SPD"
        for n in [2, 5, 10, 15]
            M = SymmetricPositiveDefinite(Int(n))
            println("Dimension: $(Int(n))")
            records, times = riemannian_median((M), 1000)
            if export_table
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
                    # e=maximum(first.(records[4])),
                    # f=times[4],
                    # g=minimum([r[2] for r in records[4]]),
                )
                CSV.write(
                    joinpath(
                        results_folder,
                        experiment_name *
                        "_$subexperiment_name" *
                        "-Comparisons-Subgrad.csv",
                        # "-Comparisons-Subgrad-CPPA.csv",
                    ),
                    B2;
                    append=true,
                )
            end
        end
    elseif subexperiment_name == "Hn"
        for n in [1, 2, 5, 10, 15]
            M = Hyperbolic(Int(2^n))
            records, times = riemannian_median((M), 1000)
            if export_table
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
                    # e=maximum(first.(records[4])),
                    # f=times[4],
                    # g=minimum([r[2] for r in records[4]]),
                )
                CSV.write(
                    joinpath(
                        results_folder,
                        experiment_name *
                        "_$subexperiment_name" *
                        "-Comparisons-Subgrad.csv",
                        # "-Comparisons-Subgrad-CPPA.csv",
                    ),
                    B2;
                    append=true,
                )
            end
        end
    elseif subexperiment_name == "Sn"
        for n in [1, 2, 5, 10, 15]
            M = Sphere(Int(2^n))
            records, times = riemannian_median((M), 1000)
            if export_table
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
                    # e=maximum(first.(records[4])),
                    # f=times[4],
                    # g=minimum([r[2] for r in records[4]]),
                )
                CSV.write(
                    joinpath(
                        results_folder,
                        experiment_name *
                        "_$subexperiment_name" *
                        "-Comparisons-Subgrad.csv",
                        # "-Comparisons-Subgrad-CPPA.csv",
                    ),
                    B2;
                    append=true,
                )
            end
        end
    end
end
