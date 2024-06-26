@doc raw"""
    RiemannianMeanCost{P}

A functor representing the Riemannian center of mass (or Riemannian mean) cost function.

For a given set of points ``d_1,\ldots,d_N`` this cost function is defined as

```math
f(p) = \sum_{j=i}^N d_{mathcal M}^2(d_i, p),
```

where ``d_{\mathcal M}`` is the [`distance`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.distance-Tuple{AbstractManifold,%20Any,%20Any}) on a Riemannian manifold.

# Constructor

    RiemannianMeanCost(M::AbstractManifold, data::AbstractVector{<:P}) where {P}

Initialize the cost function to a data set `data` of points on a manfiold `M`,
where each point is of type `P`.

# See also
[`RiemannianMeanGradient!!`](@ref ManoptExamples.RiemannianMeanGradient!!), [`Riemannian_mean_objective`](@ref ManoptExamples.Riemannian_mean_objective)

"""
struct RiemannianMeanCost{P,V<:AbstractVector{<:P}}
    data::V
end
RiemannianMeanCost(M::AbstractManifold, data) = RiemannianMeanCost(data)
function (rmc::RiemannianMeanCost)(M, p)
    return sum(distance(M, p, di)^2 for di in rmc.data)
end

@doc raw"""
    RiemannianMeanGradient!!{P} where P

A functor representing the Riemannian center of mass (or Riemannian mean) cost function.

For a given set of points ``d_1,\ldots,d_N`` this cost function is defined as

```math
\operatorname{grad}f(p) = \sum_{j=i}^N \log_p d_i
```

where ``d_{\mathcal M}`` is the [`distance`](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/functions.html#ManifoldsBase.distance-Tuple{AbstractManifold,%20Any,%20Any})
on a Riemannian manifold and we employ [`grad_distance`](https://juliamanifolds.github.io/ManifoldDiff.jl/stable/library.html#ManifoldDiff.grad_distance) to compute the single summands.

This functor provides both the allocating variant `grad_f(M,p)` as well as
the in-place variant `grad_f!(M, X, p)` which computes the gradient in-place of `X`.

# Constructors

    RiemannianMeanGradient!!(data::AbstractVector{P}, initial_vector::T=nothing) where {P,T}

Generate the Riemannian mean gradient based on some data points `data` an intial tangent vector `initial_vector`.
If you do not provide an initial tangent vector to allocate the intermediate storage of a tangent vector,
you can only use the allocating variant.

    RiemannianMeanGradient!!(
        M::AbstractManifold,
        data::AbstractVector{P};
        initial_vector::T=zero_vector(M, first(data)),
    ) where {P,T}

Initialize the Riemannian mean gradient, where the internal storage for tangent vectors can
be created automatically, since the Riemannian manifold `M` is provideed.

# See also
[`RiemannianMeanCost`](@ref ManoptExamples.RiemannianMeanCost), [`Riemannian_mean_objective`](@ref ManoptExamples.Riemannian_mean_objective)
"""
struct RiemannianMeanGradient!!{P,T,V<:AbstractVector{<:P}}
    data::V
    X::T
end
function RiemannianMeanGradient!!(
    M::AbstractManifold, data::V; initial_vector::T=zero_vector(M, first(data))
) where {P,T,V<:AbstractVector{<:P}}
    return RiemannianMeanGradient!!{P,T,V}(data, initial_vector)
end
function (rmg::RiemannianMeanGradient!!)(M, p)
    return sum(ManifoldDiff.grad_distance(M, di, p) for di in rmg.data)
end
function (rmg::RiemannianMeanGradient!!{T})(M, X::T, p) where {T}
    zero_vector!(M, X, p)
    for di in rmg.data
        ManifoldDiff.grad_distance!(M, rmg.X, di, p)
        X .+= rmg.X
    end
    return X
end

@doc raw"""
    Riemannian_mean_objective(data, initial_vector=nothing, evaluation=Manopt.AllocatingEvaluation())
    Riemannian_mean_objective(M, data;
    initial_vector=zero_vector(M, first(data)),
    evaluation=AllocatingEvaluton()
    )

Generate the objective for the Riemannian mean task for some given vector of
`data` points on the Riemannian manifold `M`.

# See also
[`RiemannianMeanCost`](@ref ManoptExamples.RiemannianMeanCost), [`RiemannianMeanGradient!!`](@ref ManoptExamples.RiemannianMeanGradient!!)

!!! note
    The first constructor should only be used if an additional storage of the vector is not
    feasible, since initialising the `initial_vector` to `nothing` disables the in-place variant.
    Hence the evaluation is a positional argument, since it only can be changed,
    if a vector is provided.

!!! note
   The objective is available when `Manopt.jl` is loaded.
"""
function Riemannian_mean_objective end
