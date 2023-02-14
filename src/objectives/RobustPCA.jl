@doc raw"""
    RobustPCACost{D,F}

A functor representing the Riemannian robust PCA function on the [Grassmann](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/grassmann.html)
manifold.
For some given (column) data ``X∈\mathbb R^{p\times n}`` the cost function is defined
on some ``\operatorname{Gr}(p,d)``, ``d<n`` as the sum of the distances
of the columns ``X_i`` to the subspace spanned by ``U\in\operatorname{Gr}(p,b)``
(represented as a point on the Stiefel manifold). The function reads

```math
f(U) = \frac{1}{n}\sum_{i=1}^n \lVert UU^{\mathrm{T}}X_i - X_i\rVert
```

This cost additionally provides a [Huber regularisation]() of the cost, that is
for some ``ε>0`` one use ``ℓ_ε(x) = \sqrt{x^2+ε^2} - ε`` in

```math
  f_{ε}(U) = \frac{1}{n}\sum_{i=1}^n ℓ_ε\bigl(\lVert UU^{\mathrm{T}}X_i - X_i\rVert\bigr)
```

# Constructor

    RobustPCACost(data::D, ε::F=1.0)

Initialize the robust PCA to some `data` ``X``
"""
struct RobustPCACost{D,F}
    data::D
    ε::F
    tmp::D
end
RobustPCACost(data::AbstractArray; ε=1.0) = RobustPCACost(data,ε,zero(data))
function (f::RobustPCACost)(M::Grassmann, U)
    f.temp .= U * U' * f.data .- f.data
    return mean(sqrt(sum(f.tmp .^ 2; dims=1) .+ f.ε^2) .- f.ε)
end

@doc raw"""
    RobustPCAGrad{D,V,F}

A functor representing the Riemannian robust PCA gradient on the  [Grassmann](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/grassmann.html)
manifold.
For some given (column) data ``X∈\mathbb R^{p\times n}`` the gradient of the
[`RobustPCACost`](@ref)
the cost function is defined
on some ``\operatorname{Gr}(p,d)``, ``d<n`` as the sum of the distances
of the columns ``X_i`` to the subspace spanned by ``U\in\operatorname{Gr}(p,b)``
(represented as a point on the Stiefel manifold). The function reads

```math
f(U) = \frac{1}{n}\sum_{i=1}^n \lVert UU^{\mathrm{T}}X_i - X_i\rVert
```

This cost additionally provides a [Huber regularisation]() of the cost, that is
for some ``ε>0`` one use ``ℓ_ε(x) = \sqrt{x^2+ε^2} - ε`` in

```math
  f_{ε}(U) = \frac{1}{n}\sum_{i=1}^n ℓ_ε\bigl(\lVert UU^{\mathrm{T}}X_i - X_i\rVert\bigr)
```

# Constructor

    RobustPCACost(data, ε=1.0)

Initialize the robust PCA to some `data` ``X``
"""

function grad_pca_cost(M, U, ϵ)
    UtX = U' * X
    vecs = U * UtX - X
    sqnrms = sum(vecs .^ 2; dims=1)
    G = zeros(p, d)
    for i in 1:n
        G = G + (1 / (sqrt(sqnrms[i] + ϵ^2))) * vecs[:, i] * UtX[:, i]'
    end
    G = 1 / n * G
    # Convert to Riemannian gradient
    return (I - U * U') * G
end