@doc raw"""
    RosenbrockCost

Provide the Rosenbrock function in 2D, i.e. for some ``a,b ∈ ℝ``

```math
f(\mathcal M, p) = a(p_1^2-p_2)^2 + (p_1-b)^2
```

which means that for the 2D case, the manifold ``\mathcal M`` is ignored.

See also [📖 Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) (with slightly different parameter naming).

# Constructor

f = Rosenbrock(a,b)

generates the struct/function of the Rosenbrock cost.
"""
struct RosenbrockCost{T}
    a::T
    b::T
end
function RosenbrockCost(::AbstractManifold=ManifoldsBase.DefaultManifold(); a=100.0, b=1.0)
    return RosenbrockCost{typeof(a + b)}(a, b)
end
function (f::RosenbrockCost)(M, p)
    return f.a * (p[1]^2 - p[2])^2 + (p[1] - f.b)^2
end

@doc raw"""
    minimizer(::RosenbrockCost)

Return the minimizer of the [`RosenbrockCost`](@ref), which is given by

```math
p^* = \begin{pmatrix} b\\b^2 \end{pmatrix}
```
"""
minimizer(f::RosenbrockCost) = [f.b, f.b^2]

@doc raw"""
    RosenbrockGradient

Provide Eclidean GRadient fo the Rosenbrock function in 2D, i.e. for some ``a,b ∈ ℝ``

```math
\nabla f(\mathcal M, p) = \begin{pmatrix}
    4a(p_1^2-p_2)p_1 + 2(p_1-b) \\
    -2a(p_1^2-p_2)
\end{pmatrix}
```

i.e. also here the manifold is ignored.

# Constructor

    RosenbrockGradient(a,b)

# Functors

    grad_f!!(M,p)
    grad_f!!(M, X, p)

evaluate the gradient at ``p`` the manifold``\mathcal M`` is ignored.
"""
struct RosenbrockGradient!!{T}
    a::T
    b::T
end
function RosenbrockGradient!!(
    ::AbstractManifold=ManifoldsBase.DefaultManifold(); a=100.0, b=1.0, kwargs...
)
    T = typeof(a + b)
    return RosenbrockGradient!!{T}(a, b)
end
function (grad_f::RosenbrockGradient!!)(M, p)
    X = zero_vector(M, p)
    grad_f(M, X, p)
    return X
end
function (grad_f::RosenbrockGradient!!)(M, X, p)
    X[1] = 4 * grad_f.a * p[1] * (p[1]^2 - p[2]) + 2 * (p[1] - grad_f.b)
    X[2] = -2 * grad_f.a * (p[1]^2 - p[2])
    return X
end

"""
    Rosenbrock_objective(M::AbstractManifold=DefaultManifold(), a=100.0, b=1.0, evaluation=AllocatingEvaluation())

Return the gradient objective of the Rosenbrock example.

See also [`RosenbrockCost`](@ref), [`RosenbrockGradient!!`](@ref)
!!! note
   The objective is available when `Manopt.jl` is loaded.
"""
function Rosenbrock_objective end

#
# A new metric
#
@doc raw"""
    RosenbrockMetric <: AbstractMetric

A metric related to the Rosenbrock problem, where the metric at a point ``p∈\mathbb R^2``
is given by


```math
⟨X,Y⟩_{\mathrm{Rb},p} = X^\mathrm{T}G_pY, \qquad
G_p = \begin{pmatrix}
  1+4p_{1}^2 & -2p_{1} \\
  -2p_{1} & 1
\end{pmatrix},
```
where the ``\mathrm{Rb}`` stands for Rosenbrock
"""
struct RosenbrockMetric <: AbstractMetric end

@doc raw"""
    Y = change_representer(M::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, ::EuclideanMetric, p, X)
    change_representer!(M::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, Y, ::EuclideanMetric, p, X)

Given the Euclidan gradient `X` at `p`, this function computes the corresponting Riesz representer `Y``
such that ``⟨X,Z⟩ = ⟨ Y, Z ⟩_{\mathrm{Rb},p}`` holds for all ``Z``, in other words ``Y = G(p)^{-1}X``.

this function is used in `riemannian_gradient` to convert a Euclidean into a Riemannian gradient.
"""
change_representer(
    ::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    ::EuclideanMetric,
    p,
    X,
)
function change_representer!(
    M::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    Y,
    ::EuclideanMetric,
    p,
    X,
)
    Y .= inverse_local_metric(M, p) * X
    return Y
end

@doc raw"""
    q = exp(::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, p, X)
    exp!(::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, q, p, X)

Compute the exponential map with respect to the [`RosenbrockMetric`](@ref ManoptExamples.RosenbrockMetric).

```math
    q = \begin{pmatrix} p_1 + X_1 \\ p_2+X_2+X_1^2\end{pmatrix}
```
"""
function expt(
    ::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    p,
    X,
    t::Number,
)
    return [p[1] + t * X[1], p[2] + t * (X[2] + X[1]^2)]
end
function exp(
    M::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    p,
    X,
)
    return ManifoldsBase.expt(M, p, X, 1.0)
end
function ManifoldsBase.expt!(
    ::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    q,
    p,
    X,
    t::Number,
)
    q[1] = p[1] + t * X[1]
    q[2] = p[2] + t * (X[2] + X[1]^2)
    return q
end
# Overwrite default to not dispatch on ODESolver
function exp!(
    M::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    q,
    p,
    X,
)
    Manifolds.expt!(M, q, p, X, 1.0)
    return q
end

@doc raw"""

    inverse_local_metric(::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, p)

Return the inverse of the local metric matrix of the [`RosenbrockMetric`](@ref ManoptExamples.RosenbrockMetric)
in the canonical unit vector basis of the tangent space ``T_p\mathbb R^2`` given as

```math
G^{-1}_p =
\begin{pmatrix}
    1 & 2p_1\\
    2p_1 & 1+4p_1^2 \\
\end{pmatrix}.
```
"""
function inverse_local_metric(
    ::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    p,
)
    return [(one(p[1])) (2*p[1]); (2*p[1]) (1+4 * p[1]^2)]
end

@doc raw"""
    inner(M::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, p, X, Y)

Compute the inner product on ``\mathbb R^2`` with respect to the [`RosenbrockMetric`](@ref RosenbrockMetric), i.e.
for ``X,Y \in T_p\mathcal M`` we have

```math
⟨X,Y⟩_{\mathrm{Rb},p} = X^\mathrm{T}G_pY, \qquad
G_p = \begin{pmatrix}
  1+4p_1^2 & -2p_1\\
  -2p_1 & 1
\end{pmatrix},
```
"""
function inner(
    M::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    p,
    X,
    Y,
)
    return transpose(X) * local_metric(M, p) * Y
end

@doc raw"""
    local_metric(::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, p)

Return the local metric matrix of the [`RosenbrockMetric`](@ref ManoptExamples.RosenbrockMetric) in the canonical unit vector basis of the tangent space ``T_p\mathbb R^2``
given as

```math
G_p = \begin{pmatrix}
  1+4p_1^2 & -2p_1 \\
  -2p_1 & 1
\end{pmatrix}
```
"""
function local_metric(
    ::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    p,
)
    return [(1+4 * p[1]^2) (-2*p[1]); (-2*p[1]) (one(p[1]))]
end

@doc raw"""
    X = log(::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, p, q)
    log!(::MetricManifold{ℝ,Euclidean{Tuple{2},ℝ},RosenbrockMetric}, X, p, q)

Compute the logarithmic map with respect to the [`RosenbrockMetric`](@ref ManoptExamples.RosenbrockMetric).
The formula reads for any ``j ∈ \{1,…,m\}``

```math
X = \begin{pmatrix}
  q_1 - p_1 \\
  q_2 - p_2 + (q_1 - p_1)^2
\end{pmatrix}
```
"""
log(
    ::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    p,
    q,
)
# note that the paper seems to have a typo here mixing up p and q
function log!(
    ::MetricManifold{
        ℝ,<:Euclidean{<:Union{TypeParameter{Tuple{2}},Tuple{<:Int}},ℝ},RosenbrockMetric
    },
    X,
    p,
    q,
)
    X[1] = q[1] - p[1]
    X[2] = q[2] - p[2] - X[1]^2
    return X
end
