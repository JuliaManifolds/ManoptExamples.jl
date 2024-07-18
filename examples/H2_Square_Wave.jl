#
# Denoising a square wave on the hyperbolic space H^2 with first order TV
# comparing the convex bundle method to
# 1. the proximal bundle method
# 2. the subgradient method
# 3. CPPA
#
using BenchmarkTools
using CSV, DataFrames
using ColorSchemes, Plots#; pgfplotsx()#PGFPlotsX
using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples
#
# Settings
experiment_name = "H2_Square_Wave_TV"
results_folder = joinpath(@__DIR__, experiment_name)
benchmarking = false
export_orig = true
export_result = true
export_table = false
toggle_debug = false
!isdir(results_folder) && mkdir(results_folder)
#
# Experiment parameters
Random.seed!(33)
n = 496
σ = 0.01 # Noise parameter
α = 0.02 # TV parameter
atol = 1e-8 # √eps()
k_max = 0.0
max_iters = 15000
# Algorithm parameters (currently not used, in favor of defaults)
# ε = Inf # Parameter of the proximal bundle method tied to the injectivity radius of the manifold
# δ = 0.01 # Update parameter for μ
# μ = 0.5 # Initial proximal parameter for the proximal bundle method
#
# Colors
signal_color = RGBA{Float64}(colorant"#BBBBBB")
noise_color = RGBA{Float64}(colorant"#33BBEE") # Tol Vibrant Teal
result_color = RGBA{Float64}(colorant"#EE7733") # Tol Vibrant Orange
#
# Functions
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
    #! In order to have length(data) == pts, we need typeof(l) == Int and mod(pts, l) == 0.
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
#
# Manifolds and data
H = Hyperbolic(2)
signal, geodesics = artificial_H2_signal(n; a=-6.0, b=6.0, T=3)
Hn = PowerManifold(H, NestedPowerRepresentation(), length(signal))
noise = map(p -> exp(H, p, rand(H; vector_at=p, σ=σ)), signal)
p0 = noise
data = noise
diameter = floatmax(Float64)
domf(M, p) = distance(M, p, p0) ≤ diameter / 2 ? true : false
#
# Objective, subgradient and prox
function f(M, p)
    return 1 / length(data) *
           (1 / 2 * distance(M, data, p)^2 + α * ManoptExamples.Total_Variation(M, p))
end
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
#
# Plot initial values
ball_scene = plot()
if export_orig
    ball_signal = convert.(PoincareBallPoint, signal)
    ball_noise = convert.(PoincareBallPoint, noise)
    ball_geodesics = convert.(PoincareBallPoint, geodesics)
    plot!(ball_scene, H, ball_signal; geodesic_interpolation=100, label="Geodesics")#, tex_output_standalone = true)
    plot!(
        ball_scene,
        H,
        ball_signal;
        markercolor=signal_color,
        markerstrokecolor=signal_color,
        label="Signal",
    )
    # savefig(ball_scene, joinpath(results_folder, experiment_name * "-signal.tex"))
    plot!(
        ball_scene,
        H,
        ball_noise;
        markercolor=noise_color,
        markerstrokecolor=noise_color,
        label="Noise",
    )
    matrix_signal = matrixify_Poincare_ball(ball_signal)
    matrix_noise = matrixify_Poincare_ball(ball_noise)
    matrix_geodesics = matrixify_Poincare_ball(ball_geodesics)
    # CSV.write(
    #     joinpath(results_folder, experiment_name * "-data.csv"),
    #     DataFrame(matrix_data, :auto);
    #     header=["x", "y"],
    # )
    # CSV.write(
    #     joinpath(results_folder, experiment_name * "-noise.csv"),
    #     DataFrame(matrix_noise, :auto);
    #     header=["x", "y"],
    # )
    # CSV.write(
    #     joinpath(results_folder, experiment_name * "-geodesics.csv"),
    #     DataFrame(matrix_geodesics, :auto);
    #     header=["x", "y"],
    # )
end
#
# Optimization
b = convex_bundle_method(
    Hn,
    f,
    ∂f,
    p0;
    # atol_λ=atol,
    # atol_errors=atol,
    # bundle_cap=50,
    domain=domf,
    k_max=k_max,
    count=[:Cost, :SubGradient],
    cache=(:LRU, [:Cost, :SubGradient], 50),
    diameter=diameter,
    stopping_criterion=StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iters),
    debug=[
        :Iteration,
        (:Cost, "F(p): %1.8f "),
        (:ξ, "ξ: %1.8f "),
        (:ε, "ε: %1.8f "),
        (:Stepsize, "stepsize: %1.4f "),
        :WarnBundle,
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
    Hn,
    f,
    ∂f,
    p0;
    count=[:Cost, :SubGradient],
    cache=(:LRU, [:Cost, :SubGradient], 50),
    stopping_criterion=StopWhenLagrangeMultiplierLess(atol) | StopAfterIteration(max_iters),
    debug=[
        :Iteration,
        :Stop,
        (:Cost, "F(p): %1.8f "),
        (:ν, "ν: %1.8f "),
        (:c, "c: %1.8f "),
        (:μ, "μ: %1.8f "),
        :Stop,
        1000,
        "\n",
    ],
    record=[:Iteration, :Cost, :Iterate],
    return_state=true,
    return_options=true,
    return_objective=true,
)
p_result = get_solver_result(p)
p_record = get_record(p)
#
s = subgradient_method(
    Hn,
    f,
    ∂f,
    p0;
    count=[:Cost, :SubGradient],
    cache=(:LRU, [:Cost, :SubGradient], 50),
    stopping_criterion=StopWhenSubgradientNormLess(√atol) | StopAfterIteration(max_iters),
    debug=[:Iteration, (:Cost, "F(p): %1.8f "), :Stop, 1000, "\n"],
    record=[:Iteration, :Cost, :Iterate],
    return_state=true,
    return_options=true,
)
s_result = get_solver_result(s)
s_record = get_record(s)
#
c = cyclic_proximal_point(
    Hn,
    f,
    proxes,
    p0;
    stopping_criterion=StopWhenAny(
        StopAfterIteration(max_iters), StopWhenChangeLess(10.0^-8)
    ),
    debug=[
        :Iteration,
        " | ",
        DebugProximalParameter(),
        " | ",
        (:Cost, "F(p): %1.8f "),
        " | ",
        :Change,
        "\n",
        1000,
        :Stop,
    ],
    record=[:Iteration, :Cost, :Iterate],
    return_state=true,
    return_options=true,
)
c_result = get_solver_result(c)
c_record = get_record(c)
#
# Results
if export_result
    # Convert hyperboloid points to Poincaré ball points
    ball_b = convert.(PoincareBallPoint, b_result)
    # ball_p = convert.(PoincareBallPoint, p_result)
    # ball_s = convert.(PoincareBallPoint, s_result)
    # ball_c = convert.(PoincareBallPoint, c_result)
    # Plot results
    plot!(
        ball_scene,
        H,
        ball_b;
        markercolor=result_color,
        markerstrokecolor=result_color,
        label="Convex Bundle Method",
    )
    # plot!(ball_scene, H, ball_p; label="Proximal Bundle Method")
    # plot!(ball_scene, H, ball_s; label="Subgradient Method")
    # plot!(ball_scene, H, ball_c; label="CPPA")
    display(ball_scene)

    matrix_b = matrixify_Poincare_ball(ball_b)
    # CSV.write(
    #     joinpath(results_folder, experiment_name * "-bundle_optimum.csv"),
    #     DataFrame(matrix_b, :auto);
    #     header=["x", "y"],
    # )
end
#
# Benchmarking
if benchmarking
    p_bm = @benchmark proximal_bundle_method(
        $Hn,
        $f,
        $∂f,
        $p0;
        cache=(:LRU, [:Cost, :SubGradient], 50),
        stopping_criterion=StopWhenLagrangeMultiplierLess($atol) |
                           StopAfterIteration($max_iters),
    )
    b_bm = @benchmark convex_bundle_method(
        $Hn,
        $f,
        $∂f,
        $p0;
        cache=(:LRU, [:Cost, :SubGradient], 50),
        diameter=$diameter,
        domain=$domf,
        k_max=$k_max,
    )
    s_bm = @benchmark subgradient_method(
        $Hn,
        $f,
        $∂f,
        $p0;
        count=[:Cost, :SubGradient],
        cache=(:LRU, [:Cost, :SubGradient], 50),
        stopping_criterion=StopWhenSubgradientNormLess(√$atol) |
                           StopAfterIteration($max_iters),
    )
    c_bm = @benchmark cyclic_proximal_point(
        $Hn,
        $f,
        $proxes,
        $p0;
        stopping_criterion=StopWhenAny(
            StopAfterIteration($max_iters), StopWhenChangeLess(10.0^-8)
        ),
    )
    #
    experiments = ["RCBM", "PBA", "SGM", "CPPA"]
    records = [b_record, p_record, s_record, c_record]
    results = [b_result, p_result, s_result, c_result]
    times = [
        median(b_bm).time * 1e-9,
        median(p_bm).time * 1e-9,
        median(s_bm).time * 1e-9,
        median(c_bm).time * 1e-9,
    ]
    #
    # Finalize - export costs
    if export_table
        for (time, record, result, experiment) in zip(times, records, results, experiments)
            A = cat(first.(record), [r[2] for r in record]; dims=2)
            CSV.write(
                joinpath(
                    results_folder, experiment_name * "_" * experiment * "-result.csv"
                ),
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
