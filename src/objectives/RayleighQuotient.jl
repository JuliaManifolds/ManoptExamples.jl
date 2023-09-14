@doc raw"""
    RayleighQuotientCost

A functor representing the Rayleigh Quotient cost function.

Let ``A ∈ ℝ^{n×n}`` be a symmetric matrix.
Then we can specify the [Rayleigh Quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient) in two forms.
Either

```
f(p) = \frac{1}{2} p^{\mathrm{T}}Ap,\qquad p ∈ 𝕊^{n-1}.
```

or extended into the embedding as

```
f(x) = \frac{1}{2} x^{\mathrm{T}}Ax, \qquad x ∈ ℝ^n,
```

which is not the orignal Rayleigh quotient for performance reasons, but
useful if you want to use this as the Euclidean cost in the emedding of ``𝕊^{n-1}``.

# Fields
* `A` – storing the matrix internally

# Constructor

    RayleighQuotientCost(A)

Create the Rayleigh cost function.
"""
struct RayleighQuotientCost{AMT}
    A::AMT
end
function (f::RayleighQuotientCost)(::Euclidean, x)
    return 0.5 * x' * A * x
end
function (f::RayleighQuotientCost)(::Sphere, p)
    return 0.5 * p' * A * p
end

@doc raw"""
    RayleighQuotientGrad!!

A functor representing the Rayleigh Quotient gradient function.

Let ``A ∈ ℝ^{n×n}`` be a symmetric matrix.
Then we can specify the gradient of the [Rayleigh Quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient)
in two forms. Either

```
\operatorname{grad} f(p) = Ap - (p^{\mathrm{T}}Ap)*p
```

or taking the Euclidean gradient of the Rayleigh quotient on the sphere as

```
∇f(x) = Ax, \qquad x ∈ ℝ^n.
```

For details, see Example 3.62 of [@Boumal:2023].

# Fields

* `A` – storing the matrix internally

# Constructor

    RayleighQuotientGrad!!(A)

Create the Rayleigh quotient gradient function.
"""
struct RayleighQuotientGrad!!{AMT}
    A::AMT
end
function (f::RayleighQuotientGrad!!)(::Euclidean, x)
    return A * x
end
function (f::RayleighQuotientGrad!!)(::Euclidean, X, x)
    X .= A * x
    return X
end
function (f::RayleighQuotientGrad!!)(::Sphere, p)
    return A * p - (p' * A * p) * p
end
function (f::RayleighQuotientGrad!!)(::Sphere, X, p)
    X .= A * p .- (p' * A * p) * p
    return X
end

@doc raw"""
    RayleighQuotientHess!!

A functor representing the Rayleigh Quotient Hessian.

Let ``A ∈ ℝ^{n×n}`` be a symmetric matrix.
Then we can specify the Hessian of the [Rayleigh Quotient](https://en.wikipedia.org/wiki/Rayleigh_quotient)
in two forms. Either

```
\operatorname{Hess} f(p)[X] =  AX - (p^{mathr{T}}AX)p - (p^{\mathrm{T}}Ap)X
```

or taking the Euclidean Hessian of the Rayleigh quotient on the sphere as

```
∇^2f(x)[V] = AV, \qquad x ∈ ℝ^n.
```

For details, see Example 5.27 of [@Boumal:2023].


# Fields
* `A` – storing the matrix internally

# Constructor

    RayleighQuotientHess!!(A)

Create the Rayleigh quotient Hessian function.
"""
struct RayleighQuotientHess!!{AMT}
    A::AMT
end
function (f::RayleighQuotientHess!!)(::Euclidean, x, V)
    return A * V
end
function (f::RayleighQuotientHess!!)(::Euclidean, W, x, V)
    W .= A * V
    return W
end
function (f::RayleighQuotientHess!!)(::Sphere, p)
    return A * X - (p' * A * X) .* p - (p' * A * p) .* X
end
function (f::RayleighQuotientHess!!)(::Sphere, Y, p, X)
    Y .= A * X - (p' * A * X) .* p - (p' * A * p) .* X
    return Y
end
