@doc raw"""
    artificial_S2_composite_Bezier_curve()

Generate a composite Bézier curve on the [Sphere]() ``\mathbb S^2`` that was used in [Bergmann, Gousenbourger, Front. Appl. Math. Stat., 2018](@cite BergmannGousenbourger:2018).

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

function artificial_S2_lemniscate(p, t::Float64, a::Float64=π / 2.0)
    M = Sphere(2)
    tP = 2.0 * Float64(p[1] >= 0.0) - 1.0 # Take north or south pole
    base = [0.0, 0.0, tP]
    xc = a * (cos(t) / (sin(t)^2 + 1.0))
    yc = a * (cos(t) * sin(t) / (sin(t)^2 + 1.0))
    tV = vector_transport_to(M, base, [xc, yc, 0.0], p, ParallelTransport())
    return exp(M, p, tV)
end
