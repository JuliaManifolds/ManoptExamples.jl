@doc raw"""
    RayleighQuotientCost

A functor representing the Rayleigh Quotient cost function.

Let ``A ∈ ℝ^{n×n}`` be a symmetric matrix.
Then we can specify the [Rayleigh Quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient) in two forms.
Either

```math
f(p) = p^{\mathrm{T}}Ap,\qquad p ∈ 𝕊^{n-1},
```

or extended into the embedding as

```math
f(x) = x^{\mathrm{T}}Ax, \qquad x ∈ ℝ^n,
```

which is not the orignal Rayleigh quotient for performance reasons, but
useful if you want to use this as the Euclidean cost in the emedding of ``𝕊^{n-1}``.

# Fields
* `A` – storing the matrix internally

# Constructor

    RayleighQuotientCost(A)

Create the Rayleigh cost function.

# See also

[`RayleighQuotientGrad!!`](@ref), [`RayleighQuotientHess!!`](@ref)
"""
struct RayleighQuotientCost{AMT}
    A::AMT
end
function (f::RayleighQuotientCost)(::Euclidean, x)
    return x' * f.A * x
end
function (f::RayleighQuotientCost)(::Sphere, p)
    return p' * f.A * p
end

@doc raw"""
    RayleighQuotientGrad!!

A functor representing the Rayleigh Quotient gradient function.

Let ``A ∈ ℝ^{n×n}`` be a symmetric matrix.
Then we can specify the gradient of the [Rayleigh Quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient)
in two forms. Either

```math
\operatorname{grad} f(p) = 2 Ap - 2 (p^{\mathrm{T}}Ap)*p,\qquad p ∈ 𝕊^{n-1},
```

or taking the Euclidean gradient of the Rayleigh quotient on the sphere as

```math
∇f(x) = 2Ax, \qquad x ∈ ℝ^n.
```

For details, see Example 3.62 of [Boumal:2023](@cite).

# Fields

* `A` – storing the matrix internally

# Constructor

    RayleighQuotientGrad!!(A)

Create the Rayleigh quotient gradient function.

# See also

[`RayleighQuotientCost`](@ref), [`RayleighQuotientHess!!`](@ref)
"""
struct RayleighQuotientGrad!!{AMT}
    A::AMT
end
function (f::RayleighQuotientGrad!!)(::Euclidean, x)
    return 2 .* f.A * x
end
function (f::RayleighQuotientGrad!!)(::Euclidean, X, x)
    X .= 2 .* f.A * x
    return X
end
function (f::RayleighQuotientGrad!!)(::Sphere, p)
    return 2 .* (f.A * p .- (p' * f.A * p) .* p)
end
function (f::RayleighQuotientGrad!!)(::Sphere, X, p)
    X .= 2 .* (f.A * p .- (p' * f.A * p) .* p)
    return X
end

@doc raw"""
    RayleighQuotientHess!!

A functor representing the Rayleigh Quotient Hessian.

Let ``A ∈ ℝ^{n×n}`` be a symmetric matrix.
Then we can specify the Hessian of the [Rayleigh Quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient)
in two forms. Either

```math
\operatorname{Hess} f(p)[X] = 2 \bigl(AX - (p^{mathrm{T}}AX)p - (p^{\mathrm{T}}Ap)X\bigr),\qquad p ∈ 𝕊^{n-1}, X \in T_p𝕊^{n-1}
```

or taking the Euclidean Hessian of the Rayleigh quotient on the sphere as

```math
∇^2f(x)[V] = 2AV, \qquad x, V ∈ ℝ^n.
```

For details, see Example 5.27 of [Boumal:2023](@cite).


# Fields
* `A` – storing the matrix internally

# Constructor

    RayleighQuotientHess!!(A)

Create the Rayleigh quotient Hessian function.

# See also

[`RayleighQuotientCost`](@ref), [`RayleighQuotientGrad!!`](@ref)
"""
struct RayleighQuotientHess!!{AMT}
    A::AMT
end
function (f::RayleighQuotientHess!!)(::Euclidean, x, V)
    return 2 .* f.A * V
end
function (f::RayleighQuotientHess!!)(::Euclidean, W, x, V)
    W .= 2 .* f.A * V
    return W
end
function (f::RayleighQuotientHess!!)(::Sphere, p, X)
    return 2 .* (f.A * X - (p' * f.A * X) .* p - (p' * f.A * p) .* X)
end
function (f::RayleighQuotientHess!!)(::Sphere, Y, p, X)
    Y .= 2 .* (f.A * X - (p' * f.A * X) .* p - (p' * f.A * p) .* X)
    return Y
end
