using QuadraticModels, RipQP
using Random, LinearAlgebra, LRUCache
using ManifoldDiff, Manifolds, Manopt, ManoptExamples

function close_point(M, p, tol; boundary=false)
    if boundary
        X = rand(M; vector_at = p)
        X .= tol * X / norm(M, p, X)
    else
        X = rand(M; vector_at = p)
        X .= tol * rand() * X / norm(M, p, X)
    end
    return retract(M, p, X, Manifolds.default_retraction_method(M, typeof(p)))
end
function outer_point(M, p, tol)
    X = rand(M; vector_at = p)
    X .= tol * (1+rand()) * X / norm(M, p, X)
    return retract(M, p, X, Manifolds.default_retraction_method(M, typeof(p)))
end
Random.seed!(30)
atol = 1e-8 # √eps()
n = 100

M = Sphere(Int(2^1))
k_min = k_max = 1.0
diam = π/6
R = diam/2
north = [0.0 for _ in 1:manifold_dimension(M)]
push!(north, 1.0)
# X = rand(M; vector_at = north)
# X /= norm(M, north, X)
# q = retract(M, north, diam * X) #[0.0, 1.0, 0.0]
data = [close_point(M, north, R) for _ in 1:n]
# pts = [rand(M) for _ in 1:10000*n]
# data = [x for x in pts if distance(M, north, x) ≤ R]
dists = [distance(M, z, y) for z in data, y in data]
#p0 = data[minimum(Tuple(findmax(dists)[2]))] #close_point(M, north, R-1e-3; boundary=true) # data[minimum(Tuple(findmax(dists)[2]))]
p0 = north

# M = Hyperbolic(4)
# k_min = k_max = -1.0
# R = 1e8#Inf
# q0 = rand(M)
# data = [rand(M) for _ in 1:n*10]
# dists = [distance(M, z, y) for z in data, y in data]
# diam = 2*R#maximum(dists)
# p0 = data[minimum(Tuple(findmax(dists)[2]))]


# Objective and subdifferential
dom(M, p) = distance(M, p, p0) ≤ R ? true : false#< R ? true : false
f(M, p) = sum(1 / length(data) * distance.(Ref(M), Ref(p), data))# : Inf
function ∂f(M, p)
    # # return zeros(manifold_dimension(M))
    # if distance(M, p, north) ≥ R - atol
    #     return sum(
    #         1 / length(data) *
    #         ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=atol),
    #     ) - rand()*log(M, p, north)
    # else#if distance(M, p, north) <  R + atol
        return sum(
            1 / length(data) *
            ManifoldDiff.subgrad_distance.(Ref(M), data, Ref(p), 1; atol=atol),
        )
    # end
end

@time b = convex_bundle_method(
    M,
    f,
    ∂f,
    p0;
    # atol_λ = atol,
    # atol_errors = atol,
    # bundle_cap=2000,
    diameter = 2R,
    domain = dom,
    k_min = k_min,
    k_max = k_max,
    count=[:Cost, :SubGradient],
    cache=(:LRU, [:Cost, :SubGradient], 50),
    debug=[
        :Iteration,
        (:Cost, "F(p): %1.16f "),
        (:ξ, "ξ: %1.8f "),
        (:ε, "ε, %1.8f "),
        (:ϱ, "ϱ: %1.4f "),
        (:diameter, "diam: %1.4f "),
        (:Stepsize, "stepsize: %1.4f "),
        # :Iterate,
        :Stop,
        1,
        "\n",
    ],
    stopping_criterion=StopAfterIteration(5000) | StopWhenLagrangeMultiplierLess(1e-8)
)

# s = subgradient_method(
#     M,
#     f,
#     ∂f,
#     p0;
#     count=[:Cost, :SubGradient],
#     cache=(:LRU, [:Cost, :SubGradient], 50),
#     stepsize=DecreasingStepsize(1, 1, 0, 1, 0, :absolute),
#     stopping_criterion=StopWhenSubgradientNormLess(√atol) | StopAfterIteration(5000),
#     debug=[
#     :Iteration,
#     (:Cost, "F(p): %1.16f, "),
#     :Stop,
#     1000,
#     "\n",
#     ],  
# )