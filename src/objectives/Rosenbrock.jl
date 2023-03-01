@doc raw"""
    RosenbrockCost

Provide the Rosenbrock function in 2D, i.e. for some ``a,b ∈ ℝ``

```math
f(\mathcal M, p) = a(p_1^2-p_2)^2 + (p_1-b)^2
```

which means that for the 2D case, the manifold ``\mathcal M`` is ignored.

See also [Rosenbrock (Wikipedia)](https://en.wikipedia.org/wiki/Rosenbrock_function)

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

    \operatorname{grad} f(M,p)
    \operatorname{grad} f(M, X, p)

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
function (f::RosenbrockGradient!!)(M, p)
    X = zero_vector(M,p)
    f(M, X, p)
    return X
end
function (f::RosenbrockGradient!!)(M, X, p)
	X[1] = 4*f.a*p[1]*(p[1]^2-p[2]) + 2*(p[1]-f.b)
	X[2] = -2*f.a*(p[1]^2-p[2])
	return X
end

"""
    Rosenbrock_objective(M::AbstractManifold=DefaultManifold(), a=100.0, b=1.0, evaluation=AllocatingEvaluation())

Return the gradient objective of the Rosenbrock example.

See also [`RosenbrockCost`](@ref), [`RosenbrockGradient!!`](@ref)
"""
function Rosenbrock_objective(
    M::AbstractManifold=ManifoldsBase.DefaultManifold();
    a=100.0,
    b=1.0,
    evaluation=AllocatingEvaluation(),
)
    return Manopt.ManifoldGradientObjective(
        RosenbrockCost(M; a=a, b=b),
        RosenbrockGradient!!(M; a=a, b=b, evaluation=evaluation);
        evaluation=evaluation,
    )
end
