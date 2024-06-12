
@doc raw"""
    artificial_S1_slope_signal([pts=500, slope=4.])

Creates a Signal of (phase-valued) data represented on the
[`Circle`](hhttps://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/circle.html) with increasing slope.

# Optional
* `pts`:    (`500`) number of points to sample the function.
* `slope`:  (`4.0`) initial slope that gets increased afterwards

This data set was introduced for the numerical examples in [BergmannLausSteidlWeinmann:2014:1](@cite)


"""
function artificial_S1_slope_signal(pts::Integer=500, slope::Float64=4.0)
    t = range(0.0, 1.0; length=pts)
    f = zero(t)
    f[t .<= 1 / 6] .= -π / 2 .+ slope * π / 8 * t[t .<= 1 / 6]
    # In the following terms, the first max
    f[(1 / 6 .< t) .& (t .<= 1 / 3)] .=
        max(f[f .!= 0]...) .- slope * π / 4 * 1 / 6 .+
        slope * π / 4 .* t[(1 / 6 .< t) .& (t .<= 1 / 3)]
    f[(1 / 3 .< t) .& (t .<= 1 / 2)] .=
        max(f[f .!= 0]...) .- slope * π / 2 * 1 / 3 .+
        slope * π / 2 * t[(1 / 3 .< t) .& (t .<= 1 / 2)]
    f[(1 / 2 .< t) .& (t .<= 2 / 3)] .=
        max(f[f .!= 0]...) .- slope * π * 1 / 2 .+
        slope * π * t[(1 / 2 .< t) .& (t .<= 2 / 3)]
    f[(2 / 3 .< t) .& (t .<= 5 / 6)] .=
        max(f[f .!= 0]...) .- slope * 2 * π * 2 / 3 .+
        slope * 2 * π * t[(2 / 3 .< t) .& (t .<= 5 / 6)]
    f[5 / 6 .< t] .=
        max(f[f .!= 0]...) .- slope * 4 * π * 5 / 6 .+ slope * 4 * π * t[5 / 6 .< t]
    return mod.(f .+ Float64(π), Ref(2 * π)) .- Float64(π)
end

@doc raw"""
    artificial_S1_signal([pts=500])

generate a real-valued signal having piecewise constant, linear and quadratic
intervals with jumps in between. If the resulting manifold the data lives on,
is the [`Circle`](hhttps://juliamanifolds.github.io/Manifolds.jl/latest/manifolds/circle.html)
the data is also wrapped to ``[BergmannLausSteidlWeinmann:2014:1](@cite).

# Optional

* `pts`: (`500`) number of points to sample the function
"""
function artificial_S1_signal(pts::Integer=500)
    t = range(0.0, 1.0; length=pts)
    f = artificial_S1_signal.(t)
    return mod.(f .+ Float64(π), Ref(2 * π)) .- Float64(π)
end
@doc raw"""
    artificial_S1_signal(x)
evaluate the example signal ``f(x), x ∈  [0,1]``,
of phase-valued data introduces in Sec. 5.1 of  [BergmannLausSteidlWeinmann:2014:1](@cite)
for values outside that interval, this Signal is `missing`.
"""
function artificial_S1_signal(x::Real)
    if x < 0
        y = missing
    elseif x <= 1 / 4
        y = -24 * π * (x - 1 / 4)^2 + 3 / 4 * π
    elseif x <= 3 / 8
        y = 4 * π * x - π / 4
    elseif x <= 1 / 2
        y = -π * x - 3 * π / 8
    elseif x <= (3 * 0 + 19) / 32
        y = -(0 + 7) / 8 * π
    elseif x <= (3 * 1 + 19) / 32
        y = -(1 + 7) / 8 * π
    elseif x <= (3 * 2 + 19) / 32
        y = -(2 + 7) / 8 * π
    elseif x <= (3 * 3 + 19) / 32
        y = -(3 + 7) / 8 * π
    elseif x <= 1
        y = 3 / 2 * π * exp(8 - 1 / (1 - x)) - 3 / 4 * π
    else
        y = missing
    end
    return y
end

@doc raw"""
    artificial_S2_composite_Bezier_curve()

Generate a composite Bézier curve on the [BergmannGousenbourger:2018](@cite).

It consists of 4 egments connecting the points
```math
\mathbf d_0 = \begin{pmatrix} 0\\0\\1\end{pmatrix},\quad
\mathbf d_1 = \begin{pmatrix} 0\\-1\\0\end{pmatrix},\quad
\mathbf d_2 = \begin{pmatrix} -1\\0\\0\end{pmatrix},\text{ and }
\mathbf d_3 = \begin{pmatrix} 0\\0\\-1\end{pmatrix}.
```

where instead of providing the two center control points explicitly we provide them as
velocities from the corresponding points, such thtat we can directly define the curve to be ``C^1``.

We define
```math
X_0 = \frac{π}{8\sqrt{2}}\begin{pmatrix}1\\-1\\0\end{pmatrix},\quad
X_1 = \frac{π}{4\sqrt{2}}\begin{pmatrix}1\\0\\1\end{pmatrix},\quad
X_2 = \frac{π}{4\sqrt{2}}\begin{pmatrix}0\\1\\-1\end{pmatrix},\text{ and }
X_3 = \frac{π}{8\sqrt{2}}\begin{pmatrix}-1\\1\\0\end{pmatrix},
```
where we defined each ``X_i \in T_{d_i}\mathbb S^2``. We defined three [`BezierSegment`](@ref)s

of cubic Bézier curves as follows
```math
\begin{align*}
b_{0,0} &= d_0, \quad & b_{1,0} &= \exp_{d_0}X_0, \quad & b_{2,0} &= \exp_{d_1}X_1, \quad & b_{3,0} &= d_1\\
b_{0,1} &= d_1, \quad & b_{1,1} &= \exp_{d_1}(-X_1), \quad & b_{2,1} &= \exp_{d_2}X_2, \quad & b_{3,1} &= d_2\\
b_{0,2} &= d_2, \quad & b_{1,1} &= \exp_{d_2}(-X_2), \quad & b_{2,2} &= \exp_{d_3}X_3, \quad & b_{3,2} &= d_3.
\end{align*}
```
"""
function artificial_S2_composite_Bezier_curve()
    M = Sphere(2)
    d0 = [0.0, 0.0, 1.0]
    d1 = [0.0, -1.0, 0.0]
    d2 = [-1.0, 0.0, 0.0]
    d3 = [0.0, 0.0, -1.0]
    #
    # control points - where b1- and b2- are constructed by the C1 condition
    #
    # We define three segments: 1
    b00 = d0 # also known as p0
    X0 = π / (8.0 * sqrt(2.0)) .* [1.0, -1.0, 0.0] # staring direction from d0
    b01 = exp(M, d0, X0) # b0+
    X1 = π / (4.0 * sqrt(2)) .* [1.0, 0.0, 1.0]
    # b02 or b1- and b11 or b1+ are constructed by this vector with opposing sign
    # to achieve a C1 curve
    b02 = exp(M, d1, X1)
    b03 = d1
    # 2
    b10 = d1
    b11 = exp(M, d1, -X1) # yields c1 condition
    X2 = -π / (4 * sqrt(2)) .* [0.0, 1.0, -1.0]
    b12 = exp(M, d2, X2)
    b13 = d2
    # 3
    b20 = d2
    b21 = exp(M, d2, -X2)
    X3 = π / (8.0 * sqrt(2)) .* [-1.0, 1.0, 0.0]
    b22 = exp(M, d3, X3)
    b23 = d3
    # hence the matrix of controlpoints for the curve reads
    return [
        BezierSegment([b00, b01, b02, b03]),
        BezierSegment([b10, b11, b12, b13]),
        BezierSegment([b20, b21, b22, b23]),
    ]
end

@doc raw"""
    Lemniscate(t::Float; kwargs...)
    Lemniscate(n::integer; interval=[0.0, 2π], kwargs...)

generate the [Lemniscate of Bernoulli](https://en.wikipedia.org/wiki/Lemniscate_of_Bernoulli)
as a curve on a manifold, by generating the curve emplying the keyword arguments below.

To be precise on the manifold `M` we use the tangent space at `p` and generate the curve

```math
γ(t) \frac{a}{}\sin^2(t) + 1 \begin{pmatrix} \cos(t) \\ \cos(t)\sin(t) \end{pmatrix}
```
in the plane spanned by `X` and `Y` in the tangent space. Note that this curve is ``2π``-periodic
and `a` is the _half-width_ of the curve.

To reproduce the first examples from [BacakBergmannSteidlWeinmann:2016](@cite) as a default,
on the sphere `p` defaults to the North pole.

THe second variant generates `n` points equispaced in ìnterval` and calls the first variant.

# Keywords

* `manifold` - ([`Sphere`]()`(2)`) the manifold to build the lemniscate on
* `p`        - (`[0.0, 0.0, 1.0]` on the sphere, `rand(M) else) the center point of the Lemniscate
* `a`        – (`π/2.0`) half-width of the Lemniscate
* `X`        – (`[1.0, 0.0, 0.0]` for the 2-sphere with default p, the first [`DefaultOrthonormalBasis`]()`()` vector otherwise)
  first direction for the plane to define the Lemniscate in, unit vector recommended.
* `Y`        – (`[0.0, 1.0, 0.0]` if p is the default, the second [`DefaultOrthonormalBasis`]()`()` vector otherwise)
  second direction for the plane to define the Lemniscate in, unit vector orthogonal to `X` recommended.
"""
function Lemniscate(
    t::Number;
    a=π / 2.0,
    manifold=Sphere(2),
    p=(manifold == Sphere(2)) ? [0.0, 0.0, 1.0] : rand(manifold),
    X=if ((manifold == Sphere(2)) && (p == [0.0, 0.0, 1.0]))
        [1.0, 0.0, 0.0]
    else
        get_vectors(manifold, p, get_basis(manifold, p, DefaultOrthonormalBasis()))[1]
    end,
    Y=if ((manifold == Sphere(2)) && (p == [0.0, 0.0, 1.0]))
        [0.0, 1.0, 0.0]
    else
        get_vectors(manifold, p, get_basis(manifold, p, DefaultOrthonormalBasis()))[2]
    end,
)
    Z = a * (cos(t) / (sin(t)^2 + 1.0)) * X + a * (cos(t) * sin(t) / (sin(t)^2 + 1.0)) * Y
    return exp(manifold, p, Z)
end
function Lemniscate(n::Integer; interval=[0.0, 2 * π], kwargs...)
    return map(t -> Lemniscate(t; kwargs...), range(interval[1], interval[2]; length=n))
end
