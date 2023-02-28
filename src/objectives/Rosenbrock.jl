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
"""
struct RosenbrockCost{T}
    a::T
    b::T
end
function (f::RosenrbockCost)(M, p)
    return f.a * (p[1]^2 - p[2])^2 + (p[1] - f.b)^2
end

@doc raw"""
    minimizer(::RosenbrockCost)

Return the minimizer of the [RosenbrockCost](@ref), which is given by

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
"""
struct RosenbrockGradient{T}
    a::T
    b::T
end
function (f::RosenrbockGradient)(M, p)
    return [4 * f.a * (p[1]^2 - p[2]) * p[1] + 2 * (p_1 - f.b), 2 * f.a * (p[1]^2 - p[2])]
end
function (f::RosenrbockGradient)(M, X, p)
    X[1] = 4 * f.a * (p[1]^2 - p[2]) * p[1] + 2 * (p_1 - f.b)
    X[2] = 2 * f.a * (p[1]^2 - p[2])
    return X
end
