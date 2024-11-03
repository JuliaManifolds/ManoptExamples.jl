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
export_orig = false
export_result = false
export_table = false
toggle_debug = false
!isdir(results_folder) && mkdir(results_folder)
#
# Parameters
Random.seed!(50)
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
    if split(string(M), "(")[1] == "Sphere"
        k_max = 1.0
        k_min = 1.0
        ε = 1e-2
    elseif split(string(M), "(")[1] == "Hyperbolic"
        k_max = 0.0
        k_min = -1.0
        ε = Inf
    else
        k_max = 0.0
        k_min = -1/2
        ε = Inf
    end
    #
    # Data
    if split(string(M), "(")[1] == "Sphere"
        centroid = [0.0 for _ in 1:manifold_dimension(M)]
        push!(centroid, 1.0)
        diameter = π/3 #manifold_dimension(M) < 2^10 ? π / 3 : π / 4 #* π/7
        data = [close_point(M, centroid, diameter / 2) for _ in 1:n]
        dists = [distance(M, z, y) for z in data, y in data]
        p0 = rand(data)#data[minimum(Tuple(findmax(dists)[2]))] #data[1]
    else
        data = [rand(M) for _ in 1:n]#[close_point(M, rand(M), diameter/2) for _ in 1:n]#
        dists = [distance(M, z, y) for z in data, y in data]
        diameter = 2maximum(dists) #10.0 #floatmax(Float64)
        p0 = rand(data) #data[minimum(Tuple(findmax(dists)[2]))] # data[findfirst(x -> x == diameter, dists)[1]] #data[1]
        centroid = p0 # data[minimum(Tuple(findmin(dists)[2]))] 
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
    domf(M, p) = distance(M, p, centroid) < diameter / 2 ? true : false
    # proxes = Function[
    #     (M, λ, p) -> prox_distance(M, λ / length(data), di, p, 1) for di in data
    # ]
    #
    # Optimization
    @time b = convex_bundle_method(
        M,
        f,
        ∂f,
        p0;
        diameter=diameter,
        domain=domf,
        k_max=k_max,
        k_min=k_min,
        count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
        debug=[
            :Iteration,
            (:Cost, "F(p): %1.16f "),
            (:ξ, "ξ: %1.8f "),
            (:ε, "ε: %1.8f "),
            (:last_stepsize, "step size: %1.8f "),
            (:null_stepsize, "null step size: %1.8f "),
            (:ϱ, "ϱ: %1.2f "),
            (:diameter, "diam: %1.2f"),
            :Stop,
            10,
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
    @time p = proximal_bundle_method(
        M,
        f,
        ∂f,
        p0;
        # δ=δ,
        # μ=μ,
        ε=ε,
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
    # @time s = subgradient_method(
    #     M,
    #     f,
    #     ∂f,
    #     p0;
    #     count=[:Cost, :SubGradient],
    #     cache=(:LRU, [:Cost, :SubGradient], 50),
    #     stepsize=DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
    #     stopping_criterion=StopWhenSubgradientNormLess(1e-4) | StopAfterIteration(1),
    #     debug=[:Iteration, (:Cost, "F(p): %1.16f "), :Stop, 1000, "\n"],
    #     record=[:Iteration, :Cost, :Iterate],
    #     return_state=true,
    #     return_options=true,
    # )
    # s_result = get_solver_result(s)
    # s_record = get_record(s)
    #
    # Benchmarking
    # b_bm = @benchmark convex_bundle_method(
    #     $M,
    #     $f,
    #     $∂f,
    #     $p0;
    #     # bundle_size=$bundle_size,
    #     diameter=$diameter,
    #     domain=$domf,
    #     k_max=$k_max,
    #     k_min=$k_min,
    #     # count=[:Cost, :SubGradient],
    #     cache=(:LRU, [:Cost, :SubGradient], 50),
    # )
    # p_bm = @benchmark proximal_bundle_method(
    #     $M,
    #     $f,
    #     $∂f,
    #     $p0;
    #     ε=$ε,
    #     # δ=$δ,
    #     # μ=$μ,
    #     # count=[:Cost, :SubGradient],
    #     cache=(:LRU, [:Cost, :SubGradient], 50),
    # )
    # s_bm = @benchmark subgradient_method(
    #     $M,
    #     $f,
    #     $∂f,
    #     $p0;
    #     # count=[:Cost, :SubGradient],
    #     cache=(:LRU, [:Cost, :SubGradient], 50),
    #     stepsize=DecreasingLength(; exponent=1, factor=1, subtrahend=0, length=1, shift=0, type=:absolute),
    #     stopping_criterion=StopWhenSubgradientNormLess(1e-4) | StopAfterIteration(5000),
    # )
    #
    # Results
    records = [
        # b_record,
        # p_record,
        # s_record,
    ]
    times = [
        # median(b_bm).time * 1e-9,
        # median(p_bm).time * 1e-9,
        # median(s_bm).time * 1e-9,
    ]
    println("   R = $(distance(M, b_result, p0))")
    return records, times
end
#
# Finalize - export costs
for subexperiment_name in ["SPD", "Hn", "Sn"]
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
    elseif subexperiment_name == "Sn"
        for n in [1, 2, 5, 10, 15]
            M = Sphere(Int(2^n))
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
    end
end
