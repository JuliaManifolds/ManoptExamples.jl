@doc raw"""
    RobustPCACost{D,F}

A functor representing the Riemannian robust PCA function on the [Grassmann](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/grassmann.html)
manifold.
For some given (column) data ``D∈\mathbb R^{d\times n}`` the cost function is defined
on some ``\operatorname{Gr}(d,m)``, ``m<n`` as the sum of the distances
of the columns ``D_i`` to the subspace spanned by ``p\in\operatorname{Gr}(d,m)``
(represented as a point on the Stiefel manifold). The function reads

```math
f(U) = \frac{1}{n}\sum_{i=1}^n \lVert pp^{\mathrm{T}}D_i - D_i\rVert
```

This cost additionally provides a [Huber regularisation](https://en.wikipedia.org/wiki/Huber_loss) of the cost, that is
for some ``ε>0`` one use ``ℓ_ε(x) = \sqrt{x^2+ε^2} - ε`` in

```math
f_{ε}(p) = \frac{1}{n}\sum_{i=1}^n ℓ_ε\bigl(\lVert pp^{\mathrm{T}}D_i - D_i\rVert\bigr)
```

Note that this is a mutable struct so you can adapt the ``ε`` later on.

# Constructor

    RobustPCACost(data::AbstractMatrix, ε=1.0)
    RobustPCACost(M::Grassmann, data::AbstractMatrix, ε=1.0)

Initialize the robust PCA cost to some `data` ``D``, and some regularization ``ε``.
The manifold is optional to comply with all examples, but it is not needed here to construct the cost.
"""
mutable struct RobustPCACost{D, F}
    data::D
    ε::F
    tmp::D
end
function RobustPCACost(data::AbstractMatrix, ε = 1.0)
    return RobustPCACost(data, ε, zero(data))
end
function RobustPCACost(::Grassmann{m, n}, data::AbstractMatrix, ε = 1.0) where {m, n}
    return RobustPCACost(data, ε, zero(data))
end
function (f::RobustPCACost)(::Grassmann, p)
    f.tmp .= p * p' * f.data .- f.data
    return mean(sqrt.(sum(f.tmp .^ 2; dims = 1) .+ f.ε^2) .- f.ε)
end

@doc raw"""
    RobustPCAGrad!!{D,F}

A functor representing the Riemannian robust PCA gradient on the [Grassmann](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/grassmann.html)
manifold.
For some given (column) data ``X∈\mathbb R^{p\times n}`` the gradient of the
[`RobustPCACost`](@ref) can be computed by projecting the Euclidean gradient onto
the corresponding tangent space.

Note that this is a mutable struct so you can adapt the ``ε`` later on.

# Constructor

    RobustPCAGrad!!(data, ε=1.0)
    RobustPCAGrad!!(M::Grassmannian{d,m}, data, ε=1.0; evaluation=AllocatingEvaluation())

Initialize the robust PCA cost to some `data` ``D``, and some regularization ``ε``.
The manifold is optional to comply with all examples, but it is not needed here to construct the cost.
Also the `evaluation=` keyword is present only for unification of the interfaces.
Indeed, independent of that keyword the functor always works in both variants.
"""
mutable struct RobustPCAGrad!!{D, F}
    data::D
    ε::F
    temp::D
end
function RobustPCAGrad!!(data::AbstractMatrix, ε = 1.0; kwargs...)
    return RobustPCAGrad!!(data, ε, zero(data))
end
function RobustPCAGrad!!(::Grassmann, data::AbstractMatrix, ε = 1.0; kwargs...)
    return RobustPCAGrad!!(data, ε, zero(data))
end
function (f::RobustPCAGrad!!)(M::Grassmann, p)
    return f(M, zero_vector(M, p), p)
end

function (f::RobustPCAGrad!!)(M::Grassmann, X, p)
    n = size(f.data, 2)
    f.temp .= p * p' * f.data .- f.data # vecs
    zero_vector!(M, X, p)
    for i in 1:n
        X .+=
            (1 / (sqrt(sum(f.temp[:, i] .^ 2) + f.ε^2))) .*
            (f.temp[:, i] * (p' * f.data[:, i])')
    end
    X ./= size(f.data, 2)
    project!(M, X, p, X) # Convert to Riemannian gradient
    return X
end

@doc raw"""
    robust_PCA_objective(data::AbstractMatrix, ε=1.0; evaluation=AllocatingEvaluation())
    robust_PCA_objective(M, data::AbstractMatrix, ε=1.0; evaluation=AllocatingEvaluton())

Generate the objective for the robust PCA task for some given `data` ``D`` and Huber regularization
parameter ``ε``.


# See also
[`RobustPCACost`](@ref ManoptExamples.RobustPCACost), [`RobustPCAGrad!!`](@ref ManoptExamples.RobustPCAGrad!!)

!!! note
    Since the construction is independent of the manifold, that argument is optional and
    mainly provided to comply with other objectives. Similarly, independent of the `evaluation`,
    indeed the gradient always allows for both the allocating and the in-place variant to be used,
    though that keyword is used to setup the objective.

!!! note
   The objective is available when `Manopt.jl` is loaded.
"""
function robust_PCA_objective end
