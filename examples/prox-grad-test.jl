using BenchmarkTools
using CSV, DataFrames
using ColorSchemes 
using Plots
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples

function shorten_vectors!(M, p, vectors)
    # If the i-th vector is of length greater than π/2, randomly shorten it
    # to a length between 0 and π/2 (excluded)
    for i in 1:length(vectors)
        if norm(M, p, vectors[i]) ≥ π/2
            # Randomly shorten the vector to a length between 0 and π/2
            new_length = rand() * (π/2)
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
# Estimate Lipschitz constant of the gradient of g
function estimate_lipschitz_constant(M, g, grad_g, anchor, R, N=100)
    constants = []
    for i in 1:N
        p = close_point(M, anchor, R)
        q = close_point(M, anchor, R)

        push!(constants, 2/distance(M, q, p)^2 * (g(M, q) - g(M, p) - inner(M, p, grad_g(M, p), log(M, p, q))))
    end
    return maximum(constants)
end

Random.seed!(100)
N = 1000
α = 1/2#0.5
σ = 1.0
atol = 1e-8
max_iters = 5000
result_color = RGBA{Float64}(colorant"#EE7733") # Tol Vibrant Orange

M = Grassmann(500, 5)# SymmetricPositiveDefinite(12) #  Hyperbolic(2) #
# p0 = rand(M)
p1 = rand(M)
# signal, geodesics = artificial_signal(M, 496; a=-6.0, b=6.0, T=3)
vectors = [rand(M; vector_at=p1, σ=σ) for _ in 1:N+2]
if split("$M", "(")[1] == "Sphere" || split("$M", "(")[1] == "Grassmann" || split("$M", "(")[1] == "Stiefel"
    shorten_vectors!(M, p1, vectors)
end
noise = [exp(M, p1, vectors[i]) for i in 1:N]
# noise = rand(M)
q = exp(M, p1, vectors[N+1])# rand(M)#log(4) * Matrix{Float64}(I, 2, 2) #rand(M)
p0 = exp(M, p1, vectors[N+2])

D = 2maximum([distance(M, p1, a) for a in vcat(noise, [q], [p0])])
# D = π

#
# Objective, subgradient and prox
g(M, p) = 1/2length(noise) * sum(distance.(Ref(M), noise, Ref(p)).^2) #log(det(p))^4#
h(M, p) = α * distance(M, p, q)
f(M, p) = g(M, p) + h(M, p)
grad_g(M, p) = 1/length(noise) * sum(ManifoldDiff.grad_distance.(Ref(M), noise, Ref(p), 2)) #4*log(det(p))^3*p #
prox_h(M, λ, p) = ManifoldDiff.prox_distance(M, α * λ, q, p, 1)
proxes = Function[(M, λ, p) -> ManifoldDiff.prox_distance(M, λ / length(noise), di, p, 2) for di in noise]
push!(proxes, (M, λ, p) -> ManifoldDiff.prox_distance(M, α * λ, q, p, 1))

if split("$M", "(")[1] == "Sphere"
	k_min = 1.0
	k_max = 1.0
elseif split("$M", "(")[1] == "Hyperbolic"
	k_min = -1.0
	k_max = -1.0
elseif split("$M", "(")[1] == "SymmetricPositiveDefinite"
	k_min = -0.5
	k_max = 0.0
elseif split("$M", "(")[1] == "Grassmann"
    k_min = 0.0
    k_max = 2.0
end

function sectional_curvature(M, p)
    X = rand(M; vector_at=p)
    Y = rand(M; vector_at=p)
    Y = Y - inner(M, p, X, Y) / norm(M, p, X)^2 * X
    R = riemann_tensor(M, p, X, Y, Y)
    return inner(M, p, R, X) / (norm(M, p, X)^2 * norm(M, p, Y)^2 - inner(M, p, X, Y)^2)
end
function ζ_1(k_min, dist)
    (k_min < zero(k_min)) && return sqrt(-k_min) * dist * coth(sqrt(-k_min) * dist)
    (k_min ≥ zero(k_min)) && return one(k_min)
end
function ζ_2(k_max, dist)
    (k_max ≤ zero(k_max)) && return one(k_max)
    (k_max > zero(k_max)) && return sqrt(k_max) * dist * cot(sqrt(k_max) * dist)
end
ϱ(k_min, k_max, dist) = max(ζ_1(k_min, dist) - 1, 1- (ζ_2(k_max, dist)))
ζ(k_min, k_max, dist) = max(ζ_1(k_min, dist), - (ζ_2(k_max, dist)))

# ζ_D = ζ_1(k_min, D)
# λ_D = 1/ζ_D

# π_k = π / √k_max
# function λ_δ(δ, M=M, h=h, grad_g=grad_g, p1=p1, k_max=k_max, D=D, N=100)
#     points = [close_point(M, p1, D/2) for _ in 1:N]
#     α_g = maximum([norm(M, p, grad_g(M, p)) for p in points])
#     α_1 = minimum([h(M, p) for p in points])
#     α_2 = maximum([h(M, p) for p in points])
#     π_k = π / √k_max
#     return (√(4*(α_2 - α_1)^2 + π_k^2/(2+δ)^2 * α_g^2) - 2*(α_2 - α_1))/(2*α_g^2)
# end
# δ = 0.1
# ζ_δ = ζ_2(k_max, π_k/(2+δ))
# L_g = estimate_lipschitz_constant(M, g, grad_g, p1, D/2)
# initial_stepsize = k_min ≥ 0.0 ? min(λ_δ(δ), ζ_δ/L_g) : λ_D

# @time c = cyclic_proximal_point(
#     M,
#     f,
#     proxes,
#     p0;
#     stopping_criterion=StopAfterIteration(max_iters) | StopWhenChangeLess(M, atol),
#     debug=[
#         :Iteration,
#         " | ",
#         DebugProximalParameter(),
#         " | ",
#         (:Cost, "F(p): %1.8f "),
#         " | ",
#         :Change,
#         "\n",
#         1000,
#         :Stop,
#     ],
#     record=[:Iteration, :Cost, :Iterate],
#     return_state=true,
#     # return_options=true,
# )
# c_result = get_solver_result(c)
# c_record = get_record(c)
#
# pg() = proximal_gradient_method(
@time pg = proximal_gradient_method(
        M,
        f,
        g,
        grad_g,
        prox_h,
        p0;
        # acceleration=Manopt.ProxGradAcceleration(
        #     M; 
        #     p=p0, 
        #     # β=1.0,
        #     # β=k->(k)/(k+3),#k->(1+√(1+4k^2))/2),
        # ),
        # λ=i -> 5e-1,#4e-4,#1e-3,#0.4225,#1e-3,#1 / (i),#1/i^2,
        stepsize=
            # ConstantLength(initial_stepsize),
            ProxGradBacktracking(; 
            # stop_when_stepsize_less=9e-1, 
            strategy=:nonconvex, 
            initial_stepsize=1.0),
            # 3/2*initial_stepsize),
        debug=[
            :Iteration, 
            (:Cost, "F(p): %1.8f "),
            " ", 
            (:Change, " last change: %1.8f "),
            (:last_stepsize, " last step size: %1.8f "),
            1000, 
            "\n", 
            :Stop,
        ],
        record=[:Iteration, :Cost, :Iterate],
        return_state=true,
        stopping_criterion=StopWhenGradientMappingNormLess(1e-8) |
                           StopAfterIteration(max_iters) | StopWhenCostNaN(),
)

# function wrapper!(results)
#     push!(results, pg())
# end
# results = []

# bm_results = @benchmark wrapper!($results)


pg_result = get_solver_result(pg)
pg_record = get_record(pg)

# Calculate the minimum cost for relative error
min_cost_pg = minimum(record[2] for record in pg_record)

# Create vectors for plotting
iterations = [record[1] for record in pg_record]

# Get initial error for scaling reference lines
relative_errors_pg = [max(record[2] - min_cost_pg, √eps()) for record in pg_record]
initial_error_pg = relative_errors_pg[1]

# Create reference trajectories
# O(1/√k)
ref_rate_sqrt_pg = [initial_error_pg/√k for k in iterations]
# O(1/k)
ref_rate_1_pg = [initial_error_pg/k for k in iterations]
# O(1/k²)
ref_rate_2_pg = [initial_error_pg/k^2 for k in iterations]
# O(1/2^k)
ref_rate_2k_pg = [initial_error_pg/2^k for k in iterations]

# Create the convergence plot
convergence_plot_pg = plot(
    iterations,
    relative_errors_pg;
    xscale=:log10,
    yscale=:log10,
    # xlims=(1e-2, 1e3),
    xlabel="Iteration (k)",
    ylabel="f(xₖ) - f*",
    label="Proximal Gradient",
    linewidth=2,
    color=result_color,
    marker=:none,
    grid=true,
    legend=:bottomleft
)

# Add reference lines
plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_sqrt_pg;
    linestyle=:dash,
    linewidth=1.5,
    color=:black,
    label="O(1/√k)"
)
plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_1_pg;
    linestyle=:dashdot,
    linewidth=1.5,
    color=:black,
    label="O(1/k)"
)
plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_2_pg;
    linestyle=:dot,
    linewidth=1.5,
    color=:black,
    label="O(1/k²)"
)
plot!(
    convergence_plot_pg,
    iterations,
    ref_rate_2k_pg;
    linestyle=:solid,
    linewidth=1.5,
    color=:black,
    label="O(1/2^k)"
)

# Optional: Save the plot
# savefig(convergence_plot, joinpath(results_folder, experiment_name * "-convergence.pdf"))

# using NonlinearSolve, StaticArrays, NLopt, ForwardDiff, JuMP
# function autodiff(f::Function)
#     function nlopt_fn(x::Vector, grad::Vector)
#         if length(grad) > 0
#             # Use ForwardDiff to compute the gradient. Replace with your
#             # favorite Julia automatic differentiation package.
#             ForwardDiff.gradient!(grad, f, x)
#         end
#         return f(x)
#     end
# end
# # For Stiefel _only_
# function T_inv(M, X, η, ζ, retraction_method=default_retraction_method(M))
#     Y = retract(M, X, η, retraction_method)
#     P = (I - Y*transpose(Y)) * ζ * (transpose(Y) * (X + η))

#     F(A, p) = transpose(X) * Y * A + A * transpose(Y) * X - ((transpose(Y)*ζ) * (transpose(Y)*(X + η)) + (transpose(Y)*(X + η)) * (transpose(Y)*ζ)) * transpose(Y)*X + transpose(X) * P + transpose(P) * X
#     u0 = ones(size(Y'*X)[1], size(Y'*X)[2])
#     prob = NonlinearProblem(F, u0)
#     solver = solve(prob)
#     return Y*solver.u + P
# end
# function T_inv_adj(M, X, η, ξ, retraction_method=default_retraction_method(M))
#     Y = retract(M, X, η, retraction_method)
#     F(B, p) = Y'*X*B + B*X'*Y - Y'*ξ
#     u0 = ones(size(Y'*X)[1], size(Y'*X)[2])
#     prob = NonlinearProblem(F, u0)
#     solver = solve(prob)
#     B = solver.u
#     return Y*(B*(X'*Y) * (Y'*(X + η)) + (Y'*(X + η)) * B*(X'*Y)) - (I - Y*Y')*(X*B + X*B' - ξ)*(Y'*(X + η))    
# end
# # Algorithm 2 from RPG
# function solve_prox!(M, g, grad_f, x, ηk, σ, L; retraction_method=default_retraction_method(M))
#     # ηk = copy(M, x, η0)
#     yk = copy(M, x)
#     ξk = copy(M, x, ηk)
#     N = ℝ^(size(ξk)[1], size(ξk)[2])
#     l(x, η) = inner(M, x, grad_f(M, x), η) + L/2 * norm(M, x, η)^2 + g(M, retract(M, x, η, retraction_method))
#     k = 0
#     while norm(M, yk, ξk) ≥ 0.003 && norm(M, yk, α*T_inv(M, x, ηk, ξk)) ≥ 0.003 && k ≤ 50
#         retract!(M, yk, x, ηk)
#         subcost(TM, ξ) = inner(TM.manifold, yk, T_inv_adj(TM.manifold, x, ηk, grad_f(TM.manifold, x) + L*ηk), ξ) + L/2 * norm(TM.manifold, yk, ξ)^2 + g(TM.manifold, yk + ξ) 
#         # my_objective_fn(ξ) = subcost(N, ξ)
#         # opt = NLopt.Opt(:LD_MMA, 0)
#         # NLopt.lower_bounds!(opt, [-Inf, 0.0])
#         # NLopt.xtol_rel!(opt, 1e-4)
#         # # But we wrap them in autodiff before passing to NLopt:
#         # NLopt.min_objective!(opt, autodiff(my_objective_fn))
#         # min_f, ξk, ret = NLopt.optimize(opt, [-100, 100])

#         # model = Model(NLopt.Optimizer)
#         # set_attribute(model, "algorithm", :LD_MMA)
#         # # set_attribute(model, "xtol_rel", 1e-4)
#         # # set_attribute(model, "constrtol_abs", 1e-8)
#         # @variable(model, ξ)
#         # # set_lower_bound(x[2], 0.0)
#         # # set_start_value.(x, [1.234, 5.678])
#         # @NLobjective(model, Min, my_objective_fn(ξ))
#         # JuMP.optimize!(model)
#         # min_f, ξk, ret = objective_value(model), value.(ξ), raw_status(model)
#         # println(
#         #     """
#         #     objective value       : $min_f
#         #     solution              : $ξk
#         #     solution status       : $ret
#         #     """
#         # )

#         # Subsolver's subsolver
#         ξk = NelderMead(TangentSpace(M, yk), subcost)#mesh_adaptive_direct_search(N, subcost)
#         α = 1
#         while l(x, ηk + α* T_inv(M, x, ηk, ξk)) ≥ l(x, ηk) - σ*α*norm(M, x, ξk)^2
#             α *= 0.5
#         end
#         ηk .+= α*T_inv(M, x, ηk, ξk)
#         k += 1
#     end
#     return ηk
# end
# # Algorithm 1 from RPG
# # f is the smooth component, g is the non-smooth component
# function RPG(M, f, g, grad_f, p0=rand(M); σ=0.1, L=1, retraction_method=default_retraction_method(M), return_state=true, tol=1e-8)
#     x = copy(M, p0)
#     η = rand(M; vector_at = x)
#     k = 1
#     state = []
#     n, m = Manifolds.get_parameter(M.size)
#     while norm(M, x, η)^2 * n * m ≥ tol
#         solve_prox!(M, g, grad_f, x, η, σ, L; retraction_method=retraction_method)
#         retract!(M, x, x, η, retraction_method)
#         if return_state
#             push!(state, (k, f(M, x) + g(M, x)))
#         end
#         k += 1
#     end
#     return x, k, state
# end

# @time rpg_result = RPG(
#     M,
#     g,
#     h,
#     grad_g,
#     p0;
#     # σ=0.1,
#     # L=1,
#     # retraction_method=default_retraction_method(M),
#     # return_state=true,
# )