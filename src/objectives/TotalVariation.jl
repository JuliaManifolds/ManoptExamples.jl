#
# = Total Variation cost functions =
#

# == Cost functions ==
@doc raw"""
    Intrinsic_infimal_convolution_TV12(M, f, u, v, α, β)

Compute the intrinsic infimal convolution model, where the addition is replaced
by a mid point approach and the two functions involved are [`second_order_Total_Variation`](@ref)
and [`Total_Variation`](@ref). The model reads

```math
E(u,v) =
  \frac{1}{2}\sum_{i ∈ \mathcal G}
    d_{\mathcal M}\bigl(g(\frac{1}{2},v_i,w_i),f_i\bigr)
  +\alpha\bigl( β\mathrm{TV}(v) + (1-β)\mathrm{TV}_2(w) \bigr).
```

for more details see [BergmannFitschenPerschSteidl:2017, BergmannFitschenPerschSteidl:2018](@cite).

# See also

[`Total_Variation`](@ref), [`second_order_Total_Variation`](@ref)
"""
function Intrinsic_infimal_convolution_TV12(M::AbstractManifold, f, u, v, α, β)
    IC = 1 / 2 * distance(M, shortest_geodesic(M, u, v, 0.5), f)^2
    TV12 = β * Total_Variation(M, u) + (1 - β) * second_order_Total_Variation(M, v)
    return IC + α * TV12
end

@doc raw"""
    L2_Total_Variation(M, p_data, α, p)

compute the ``ℓ^2``-TV functional on the [PowerManifold](https://juliamanifolds.github.io/ManifoldsBase.jl/stable/metamanifolds/#sec-power-manifold) `M` for given
(fixed) data `p_data` (on `M`), a nonnegative weight `α`, and evaluated at `p` (on `M`),
i.e.

```math
E(p) = d_{\mathcal M}^2(f,p) + \alpha \operatorname{TV}(p)
```

# See also

[`Total_Variation`](@ref)
"""
L2_Total_Variation(M, p_data, α, p) =
    1 / 2 * distance(M, p_data, p)^2 + α * Total_Variation(M, p)

@doc raw"""
    L2_Total_Variation_1_2(M, f, α, β, x)

compute the ``ℓ^2``-TV-TV2 functional on the `PowerManifold` manifold `M` for
given (fixed) data `f` (on `M`), nonnegative weight `α`, `β`, and evaluated
at `x` (on `M`), i.e.

```math
E(x) = d_{\mathcal M}^2(f,x) + \alpha\operatorname{TV}(x)
  + β\operatorname{TV}_2(x)
```

# See also

[`Total_Variation`](@ref), [`second_order_Total_Variation`](@ref)
"""
function L2_Total_Variation_1_2(M::PowerManifold, f, α, β, p)
    return 1 / 2 * distance(M, f, p)^2 +
           α * Total_Variation(M, p) +
           β * second_order_Total_Variation(M, p)
end

@doc raw"""
    L2_second_order_Total_Variation(M, f, β, x)

compute the ``ℓ^2``-TV2 functional on the `PowerManifold` manifold `M`
for given data `f`, nonnegative parameter `β`, and evaluated at `x`, i.e.

```math
E(x) = d_{\mathcal M}^2(f,x) + β\operatorname{TV}_2(x)
```

as used in [BacakBergmannSteidlWeinmann:2016](@cite).

# See also

[`second_order_Total_Variation`](@ref)
"""
function L2_second_order_Total_Variation(M::PowerManifold, f, β, x)
    return 1 / 2 * distance(M, f, x)^2 + β * second_order_Total_Variation(M, x)
end

@doc raw"""
    Total_Variation(M,x [,p=2,q=1])

Compute the ``\operatorname{TV}^p`` functional for data `x`on the `PowerManifold`
manifold `M`, i.e. ``\mathcal M = \mathcal N^n``, where ``n ∈ \mathbb N^k`` denotes
the dimensions of the data `x`.
Let ``\mathcal I_i`` denote the forward neighbors, i.e. with ``\mathcal G`` as all
indices from ``\mathbf{1} ∈ \mathbb N^k`` to ``n`` we have
``\mathcal I_i = \{i+e_j, j=1,…,k\}\cap \mathcal G``.
The formula reads

```math
E^q(x) = \sum_{i ∈ \mathcal G}
  \bigl( \sum_{j ∈  \mathcal I_i} d^p_{\mathcal M}(x_i,x_j) \bigr)^{q/p},
```

see [WeinmannDemaretStorath:2014](@cite) for more details.
In long function names, this might be shortened to `TV1` and the `1` might be ommitted if
only total variation is present.

# See also

[`grad_Total_Variation`](@ref), [`prox_Total_Variation`](@ref), [`second_order_Total_Variation`](@ref)
"""
function Total_Variation(M::PowerManifold, x, p=1, q=1)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R)
    cost = fill(0.0, Tuple(power_size))
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for i in R # iterate over all pixel
            j = i + ek # compute neighbor
            if all(map(<=, j.I, maxInd.I)) # is this neighbor in range?
                cost[i] += distance(M.manifold, x[M, Tuple(i)...], x[M, Tuple(j)...])^p
            end
        end
    end
    cost = (cost) .^ (1 / p)
    if q > 0
        return sum(cost .^ q)^(1 / q)
    else
        return cost
    end
end

@doc raw"""
    second_order_Total_Variation(M,(x1,x2,x3) [,p=1])

Compute the ``\operatorname{TV}_2^p`` functional for the 3-tuple of points
`(x1,x2,x3)`on the manifold `M`. Denote by

```math
  \mathcal C = \bigl\{ c ∈  \mathcal M \ |\ g(\tfrac{1}{2};x_1,x_3) \text{ for some geodesic }g\bigr\}
```

the set of mid points between ``x_1`` and ``x_3``. Then the function reads

```math
d_2^p(x_1,x_2,x_3) = \min_{c ∈ \mathcal C} d_{\mathcal M}(c,x_2),
```

see [BacakBergmannSteidlWeinmann:2016](@cite) for a derivation.
In long function names, this might be shortened to `TV2`.


# See also

[`grad_second_order_Total_Variation`](@ref), [`prox_second_order_Total_Variation`](@ref), [`Total_Variation`](@ref)
"""
function second_order_Total_Variation(
    M::MT, x::Tuple{T,T,T}, p=1
) where {MT<:AbstractManifold,T}
    # note that here mid_point returns the closest to x2 from the e midpoints between x1 x3
    return 1 / p * distance(M, mid_point(M, x[1], x[3]), x[2])^p
end
@doc raw"""
    second_order_Total_Variation(M,x [,p=1])

compute the ``\operatorname{TV}_2^p`` functional for data `x` on the
`PowerManifold` manifoldmanifold `M`, i.e. ``\mathcal M = \mathcal N^n``,
where ``n ∈ \mathbb N^k`` denotes the dimensions of the data `x`.
Let ``\mathcal I_i^{\pm}`` denote the forward and backward neighbors, respectively,
i.e. with ``\mathcal G`` as all indices from ``\mathbf{1} ∈ \mathbb N^k`` to ``n`` we
have ``\mathcal I^\pm_i = \{i\pm e_j, j=1,…,k\}\cap \mathcal I``.
The formula then reads

```math
E(x) = \sum_{i ∈ \mathcal I,\ j_1 ∈  \mathcal I^+_i,\ j_2 ∈  \mathcal I^-_i}
d^p_{\mathcal M}(c_i(x_{j_1},x_{j_2}), x_i),
```

where ``c_i(⋅,⋅)`` denotes the mid point between its two arguments that is
nearest to ``x_i``, see [BacakBergmannSteidlWeinmann:2016](@cite) for a derivation.

In long function names, this might be shortened to `TV2`.

# See also

[`grad_second_order_Total_Variation`](@ref), [`prox_second_order_Total_Variation`](@ref)
"""
function second_order_Total_Variation(M::PowerManifold, x, p::Int=1, Sum::Bool=true)
    Tt = Tuple(power_dimensions(M))
    R = CartesianIndices(Tt)
    d = length(Tt)
    minInd, maxInd = first(R), last(R)
    cost = fill(0.0, Tt)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for i in R # iterate over all pixel
            jF = i + ek # compute forward neighbor
            jB = i - ek # compute backward neighbor
            if all(map(<=, jF.I, maxInd.I)) && all(map(>=, jB.I, minInd.I)) # are neighbors in range?
                cost[i] += second_order_Total_Variation(
                    M.manifold,
                    (x[M, Tuple(jB)...], x[M, Tuple(i)...], x[M, Tuple(jF)...]),
                    p,
                )
            end
        end # i in R
    end # directions
    if p != 1
        cost = (cost) .^ (1 / p)
    end
    if Sum
        return sum(cost)
    else
        return cost
    end
end

# (b) Gradients
@doc raw"""
    grad_u, grad_v = grad_intrinsic_infimal_convolution_TV12(M, f, u, v, α, β)

compute (sub)gradient of the intrinsic infimal convolution model using the mid point
model of second order differences, see [`second_order_Total_Variation`](@ref), i.e. for some ``f ∈ \mathcal M``
on a `PowerManifold` manifold ``\mathcal M`` this function computes the (sub)gradient of

```math
E(u,v) =
\frac{1}{2}\sum_{i ∈ \mathcal G} d_{\mathcal M}(g(\frac{1}{2},v_i,w_i),f_i)
+ \alpha
\bigl(
β\mathrm{TV}(v) + (1-β)\mathrm{TV}_2(w)
\bigr),
```

where both total variations refer to the intrinsic ones, [`grad_Total_Variation`](@ref) and
[`grad_second_order_Total_Variation`](@ref), respectively.
"""
function grad_intrinsic_infimal_convolution_TV12(M::AbstractManifold, f, u, v, α, β)
    c = mid_point(M, u, v, f)
    iL = log(M, c, f)
    return adjoint_differential_shortest_geodesic_startpoint(M, u, v, 1 / 2, iL) +
           α * β * grad_Total_Variation(M, u),
    adjoint_differential_shortest_geodesic_endpoint(M, u, v, 1 / 2, iL) +
    α * (1 - β) * grad_second_order_Total_Variation(M, v)
end

@doc raw"""
    X = grad_Total_Variation(M, (x,y)[, p=1])
    grad_Total_Variation!(M, X, (x,y)[, p=1])

compute the (sub) gradient of ``\frac{1}{p}d^p_{\mathcal M}(x,y)`` with respect
to both ``x`` and ``y`` (in place of `X` and `Y`).
"""
function grad_Total_Variation(M::AbstractManifold, q::Tuple{T,T}, p=1) where {T}
    if p == 2
        return (-log(M, q[1], q[2]), -log(M, q[2], q[1]))
    else
        d = distance(M, q[1], q[2])
        if d == 0 # subdifferential containing zero
            return (zero_vector(M, q[1]), zero_vector(M, q[2]))
        else
            return (-log(M, q[1], q[2]) / (d^(2 - p)), -log(M, q[2], q[1]) / (d^(2 - p)))
        end
    end
end
function grad_Total_Variation!(M::AbstractManifold, X, q::Tuple{T,T}, p=1) where {T}
    d = distance(M, q[1], q[2])
    if d == 0 # subdifferential containing zero
        zero_vector!(M, X[1], q[1])
        zero_vector!(M, X[2], q[2])
        return X
    end
    log!(M, X[1], q[1], q[2])
    log!(M, X[2], q[2], q[1])
    if p == 2
        X[1] .*= -1
        X[2] .*= -1
    else
        X[1] .*= -1 / (d^(2 - p))
        X[2] .*= -1 / (d^(2 - p))
    end
    return X
end
@doc raw"""
    X = grad_Total_Variation(M, λ, x[, p=1])
    grad_Total_Variation!(M, X, λ, x[, p=1])

Compute the (sub)gradient ``∂f`` of all forward differences occurring,
in the power manifold array, i.e. of the function

```math
f(p) = \sum_{i}\sum_{j ∈ \mathcal I_i} d^p(x_i,x_j)
```

where ``i`` runs over all indices of the `PowerManifold` manifold `M`
and ``\mathcal I_i`` denotes the forward neighbors of ``i``.

# Input
* `M` – a `PowerManifold` manifold
* `x` – a point.

# Output
* X – resulting tangent vector in ``T_x\mathcal M``. The computation can also be done in place.
"""
function grad_Total_Variation(M::PowerManifold, x, p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R)
    X = zero_vector(M, x)
    c = Total_Variation(M, x, p, 0)
    for i in R # iterate over all pixel
        di = 0.0
        for k in 1:d # for all direction combinations
            ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
            j = i + ek # compute neighbor
            if all(map(<=, j.I, maxInd.I)) # is this neighbor in range?
                if p != 1
                    g =
                        (c[i] == 0 ? 1 : 1 / c[i]) .*
                        grad_Total_Variation(M.manifold, (x[i], x[j]), p) # Compute TV on these
                else
                    g = grad_Total_Variation(M.manifold, (x[i], x[j]), p) # Compute TV on these
                end
                X[i] += g[1]
                X[j] += g[2]
            end
        end # directions
    end # i in R
    return X
end
function grad_Total_Variation!(M::PowerManifold, X, x, p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R)
    c = Total_Variation(M, x, p, 0)
    g = [zero_vector(M.manifold, x[first(R)]), zero_vector(M.manifold, x[first(R)])]
    for i in R # iterate over all pixel
        di = 0.0
        for k in 1:d # for all direction combinations
            ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
            j = i + ek # compute neighbor
            if all(map(<=, j.I, maxInd.I)) # is this neighbor in range?
                grad_Total_Variation!(M.manifold, g, (x[i], x[j]), p) # Compute TV on these
                if p != 1
                    (c[i] != 0) && (g[1] .*= 1 / c[i])
                    (c[i] != 0) && (g[2] .*= 1 / c[i])
                end
                X[i] += g[1]
                X[j] += g[2]
            end
        end # directions
    end # i in R
    return X
end
@doc raw"""
    Y = grad_second_order_Total_Variation(M, q[, p=1])
    grad_second_order_Total_Variation!(M, Y, q[, p=1])

computes the (sub) gradient of ``\frac{1}{p}d_2^p(q_1, q_2, q_3)`` with respect
to all three components of ``q∈\mathcal M^3``, where ``d_2`` denotes the second order
absolute difference using the mid point model, i.e. let

```math
\mathcal C = \bigl\{ c ∈ \mathcal M \ |\ g(\tfrac{1}{2};q_1,q_3) \text{ for some geodesic }g\bigr\}
```
denote the mid points between ``q_1`` and ``q_3`` on the manifold ``\mathcal M``.
Then the absolute second order difference is defined as

```math
d_2(q_1,q_2,q_3) = \min_{c ∈ \mathcal C_{q_1,q_3}} d(c, q_2).
```

While the (sub)gradient with respect to ``q_2`` is easy, the other two require
the evaluation of an `adjoint_Jacobi_field`.

The derivation of this gradient can be found in [BacakBergmannSteidlWeinmann:2016](@cite).
"""
function grad_second_order_Total_Variation(M::AbstractManifold, q, p::Int=1)
    X = [zero_vector(M, x) for x in q]
    return grad_second_order_Total_Variation!(M, X, q, p)
end
function grad_second_order_Total_Variation!(M::AbstractManifold, X, q, p::Int=1)
    c = mid_point(M, q[1], q[3], q[2]) # nearest mid point of x and z to y
    d = distance(M, q[2], c)
    innerLog = -log(M, c, q[2])
    if p == 2
        X[1] .= adjoint_differential_shortest_geodesic_startpoint(
            M, q[1], q[3], 1 / 2, innerLog
        )
        log!(M, X[2], q[2], c)
        X[2] .*= -1
        X[3] .= adjoint_differential_shortest_geodesic_endpoint(
            M, q[1], q[3], 1 / 2, innerLog
        )
    else
        if d == 0 # subdifferential containing zero
            for i in 1:3
                zero_vector!(M, X[i], q[i])
            end
        else
            X[1] .= adjoint_differential_shortest_geodesic_startpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
            log!(M, X[2], q[2], c)
            X[2] .*= -1 / (d^(2 - p))
            X[3] .= adjoint_differential_shortest_geodesic_endpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
        end
    end
    return X
end
@doc raw"""
    grad_second_order_Total_Variation(M::PowerManifold, q[, p=1])

computes the (sub) gradient of ``\frac{1}{p}d_2^p(q_1,q_2,q_3)``
with respect to all ``q_1,q_2,q_3`` occurring along any array dimension in the
point `q`, where `M` is the corresponding `PowerManifold`.
"""
function grad_second_order_Total_Variation(M::PowerManifold, q, p::Int=1)
    X = zero_vector(M, q)
    return grad_second_order_Total_Variation!(M, X, q, p)
end
function grad_second_order_Total_Variation!(M::PowerManifold, X, q, p::Int=1)
    power_size = power_dimensions(M)
    rep_size = representation_size(M.manifold)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    minInd, maxInd = first(R), last(R)
    c = second_order_Total_Variation(M, q, p, false)
    for i in R # iterate over all pixel
        di = 0.0
        for k in 1:d # for all direction combinations
            ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
            jF = i + ek # compute forward neighbor
            jB = i - ek # compute backward neighbor
            if all(map(<=, jF.I, maxInd.I)) && all(map(>=, jB.I, minInd.I)) # are neighbors in range?
                if p != 1
                    g =
                        (c[i] == 0 ? 1 : 1 / c[i]) .* grad_second_order_Total_Variation(
                            M.manifold, (q[jB], q[i], q[jF]), p
                        ) # Compute TV2 on these
                else
                    g = grad_second_order_Total_Variation(
                        M.manifold, (q[jB], q[i], q[jF]), p
                    ) # Compute TV2 on these
                end
                X[M, jB.I...] = g[1]
                X[M, i.I...] = g[2]
                X[M, jF.I...] = g[3]
            end
        end # directions
    end # i in R
    return X
end

# (c) proxial maps
@doc raw"""
    [y1,y2] = prox_Total_Variation(M, λ, [x1,x2] [,p=1])
    prox_Total_Variation!(M, [y1,y2] λ, [x1,x2] [,p=1])

Compute the proximal map ``\operatorname{prox}_{λ\varphi}`` of
``φ(x,y) = d_{\mathcal M}^p(x,y)`` with
parameter `λ`.
A derivation of this closed form solution is given in see [WeinmannDemaretStorath:2014](@cite).

# Input

* `M` – a manifold `M`
* `λ` – a real value, parameter of the proximal map
* `(x1,x2)` – a tuple of two points,

# Optional

(default is given in brackets)
* `p` – (1) exponent of the distance of the TV term

# Output

* `(y1,y2)` – resulting tuple of points of the ``\operatorname{prox}_{λφ}(```(x1,x2)```)``.
  The result can also be computed in place.
"""
function prox_Total_Variation(
    M::AbstractManifold, λ::Number, x::Tuple{T,T}, p::Int=1
) where {T}
    d = distance(M, x[1], x[2])
    if p == 1
        t = min(0.5, λ / d)
    elseif p == 2
        t = λ / (1 + 2 * λ)
    else
        throw(
            ErrorException(
                "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    return (exp(M, x[1], log(M, x[1], x[2]), t), exp(M, x[2], log(M, x[2], x[1]), t))
end
function prox_Total_Variation!(
    M::AbstractManifold, y, λ::Number, x::Tuple{T,T}, p::Int=1
) where {T}
    d = distance(M, x[1], x[2])
    if p == 1
        t = min(0.5, λ / d)
    elseif p == 2
        t = λ / (1 + 2 * λ)
    else
        throw(
            ErrorException(
                "Proximal Map of TV(M,x1,x2,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
    X1 = log(M, x[1], x[2])
    X2 = log(M, x[2], x[1])
    exp!(M, y[1], x[1], X1, t)
    exp!(M, y[2], x[2], X2, t)
    return y
end
@doc raw"""
    ξ = prox_Total_Variation(M,λ,x [,p=1])

compute the proximal maps ``\operatorname{prox}_{λ\varphi}`` of
all forward differences occurring in the power manifold array, i.e.
``\varphi(xi,xj) = d_{\mathcal M}^p(xi,xj)`` with `xi` and `xj` are array
elements of `x` and `j = i+e_k`, where `e_k` is the ``k``th unit vector.
The parameter `λ` is the prox parameter.

# Input
* `M` – a manifold `M`
* `λ` – a real value, parameter of the proximal map
* `x` – a point.

# Optional
(default is given in brackets)
* `p` – (1) exponent of the distance of the TV term

# Output
* `y` – resulting  point containing with all mentioned proximal
  points evaluated (in a cyclic order). The computation can also be done in place
"""
function prox_Total_Variation(M::PowerManifold, λ, x, p::Int=1)
    y = deepcopy(x)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1
            for i in R # iterate over all pixel
                if (i[k] % 2) == l
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neighbour index as Cartesian Index
                        (y[i], y[j]) = prox_Total_Variation(M.manifold, λ, (y[i], y[j]), p) # Compute TV on these
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end
function prox_Total_Variation!(M::PowerManifold, y, λ, x, p::Int=1)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    copyto!(M, y, x)
    maxInd = last(R).I
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1
            for i in R # iterate over all pixel
                if (i[k] % 2) == l # even/odd splitting
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neighbour index as Cartesian Index
                        prox_Total_Variation!(M.manifold, [y[i], y[j]], λ, (y[i], y[j]), p) # Compute TV on these
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end

@doc raw"""
    y = prox_parallel_TV(M, λ, x [,p=1])
    prox_parallel_TV!(M, y, λ, x [,p=1])

compute the proximal maps ``\operatorname{prox}_{λφ}`` of
all forward differences occurring in the power manifold array, i.e.
``φ(x_i,x_j) = d_{\mathcal M}^p(x_i,x_j)`` with `xi` and `xj` are array
elements of `x` and `j = i+e_k`, where `e_k` is the ``k``th unit vector.
The parameter `λ` is the prox parameter.

# Input
* `M`     – a `PowerManifold` manifold
* `λ`     – a real value, parameter of the proximal map
* `x`     – a point

# Optional
(default is given in brackets)
* `p` – (`1`) exponent of the distance of the TV term

# Output
* `y`  – resulting Array of points with all mentioned proximal
  points evaluated (in a parallel within the arrays elements).
  The computation can also be done in place.

*See also* [`prox_Total_Variation`](@ref)
"""
function prox_parallel_TV(M::PowerManifold, λ, x::AbstractVector, p::Int=1)
    R = CartesianIndices(x[1])
    d = ndims(x[1])
    if length(x) != 2 * d
        throw(
            ErrorException(
                "The number of inputs from the array ($(length(x))) has to be twice the data dimensions ($(d)).",
            ),
        )
    end
    maxInd = Tuple(last(R))
    # create an array for even/odd splitted proxes along every dimension
    y = deepcopy(x)
    yV = reshape(y, d, 2)
    xV = reshape(x, d, 2)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1 # even odd
            for i in R # iterate over all pixel
                if (i[k] % 2) == l
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neighbour index as Cartesian Index
                        # parallel means we apply each (direction even/odd) to a separate copy of the data.
                        (yV[k, l + 1][i], yV[k, l + 1][j]) = prox_Total_Variation(
                            M.manifold, λ, (xV[k, l + 1][i], xV[k, l + 1][j]), p
                        ) # Compute TV on these
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end
function prox_parallel_TV!(
    M::PowerManifold, y::AbstractVector, λ, x::AbstractVector, p::Int=1
)
    R = CartesianIndices(x[1])
    d = ndims(x[1])
    if length(x) != 2 * d
        throw(
            ErrorException(
                "The number of inputs from the array ($(length(x))) has to be twice the data dimensions ($(d)).",
            ),
        )
    end
    maxInd = Tuple(last(R))
    # init y
    for i in 1:length(x)
        copyto!(M, y[i], x[i])
    end
    yV = reshape(y, d, 2)
    xV = reshape(x, d, 2)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:1 # even odd
            for i in R # iterate over all pixel
                if (i[k] % 2) == l
                    J = i.I .+ ek.I #i + e_k is j
                    if all(J .<= maxInd) # is this neighbor in range?
                        j = CartesianIndex(J...) # neighbour index as Cartesian Index
                        # parallel means we apply each (direction even/odd) to a separate copy of the data.
                        prox_Total_Variation!(
                            M.manifold,
                            [yV[k, l + 1][i], yV[k, l + 1][j]],
                            λ,
                            (xV[k, l + 1][i], xV[k, l + 1][j]),
                            p,
                        ) # Compute TV on these in place of y
                    end
                end
            end # i in R
        end # even odd
    end # directions
    return y
end

@doc raw"""
    (y1,y2,y3) = prox_second_order_Total_Variation(M,λ,(x1,x2,x3),[p=1], kwargs...)
    prox_second_order_Total_Variation!(M, y, λ,(x1,x2,x3),[p=1], kwargs...)

Compute the proximal map ``\operatorname{prox}_{λ\varphi}`` of
``\varphi(x_1,x_2,x_3) = d_{\mathcal M}^p(c(x_1,x_3),x_2)`` with
parameter `λ`>0, where ``c(x,z)`` denotes the mid point of a shortest
geodesic from `x1` to `x3` that is closest to `x2`.
The result can be computed in place of `y`.

Note that this function does not have a closed form solution but is solbed by
a few steps of the [subgradient mehtod](https://manoptjl.org/stable/solvers/subgradient/) from [manopt.jl]() by default.
See [BacakBergmannSteidlWeinmann:2016](@cite) for a derivation.

# Input

* `M`          – a manifold
* `λ`          – a real value, parameter of the proximal map
* `(x1,x2,x3)` – a tuple of three points

* `p` – (`1`) exponent of the distance of the TV term

# Optional
`kwargs...` – parameters for the internal [`subgradient_method`](https://manoptjl.org/stable/solvers/subgradient/#Manopt.subgradient_method)
    (if `M` is neither `Euclidean` nor `Circle`, since for these a closed form
    is given)

# Output
* `(y1,y2,y3)` – resulting tuple of points of the proximal map.
  The computation can also be done in place.

  !!! note
    This function rewuires `Manopt.jl` to be loaded
"""
function prox_second_order_Total_Variation end

@doc raw"""
    y = prox_second_order_Total_Variation(M, λ, x[, p=1])
    prox_second_order_Total_Variation!(M, y, λ, x[, p=1])

compute the proximal maps ``\operatorname{prox}_{λ\varphi}`` of
all centered second order differences occurring in the power manifold array, i.e.
``\varphi(x_k,x_i,x_j) = d_2(x_k,x_i.x_j)``, where ``k,j`` are backward and forward
neighbors (along any dimension in the array of `x`).
The parameter `λ` is the prox parameter.

# Input
* `M` – a manifold `M`
* `λ` – a real value, parameter of the proximal map
* `x` – a points.

# Optional
(default is given in brackets)
* `p` – (`1`) exponent of the distance of the TV term

# Output
* `y` – resulting point with all mentioned proximal points evaluated (in a cyclic order).
  The computation can also be done in place.

!!! note
    This function rewuires `Manopt.jl` to be loaded
"""
function prox_second_order_Total_Variation(
    M::PowerManifold{N,T}, λ, x, p::Int=1
) where {N,T}
    y = deepcopy(x)
    return prox_second_order_Total_Variation!(M, y, λ, x, p)
end
function prox_second_order_Total_Variation!(
    M::PowerManifold{N,T}, y, λ, x, p::Int=1
) where {N,T}
    power_size = power_dimensions(M)
    R = CartesianIndices(power_size)
    d = length(size(x))
    minInd = first(R).I
    maxInd = last(R).I
    copyto!(M, y, x)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:2
            for i in R # iterate over all pixel
                if (i[k] % 3) == l
                    JForward = i.I .+ ek.I #i + e_k
                    JBackward = i.I .- ek.I # i - e_k
                    all(JForward .<= maxInd) &&
                        all(JBackward .>= minInd) &&
                        prox_second_order_Total_Variation!(
                            M.manifold,
                            [y[M, JBackward...], y[M, i.I...], y[M, JForward...]],
                            λ,
                            (y[M, JBackward...], y[M, i.I...], y[M, JForward...]),
                            p,
                        )
                end # if mod 3
            end # i in R
        end # for mod 3
    end # directions
    return y
end
function prox_second_order_Total_Variation(
    ::Euclidean, λ, pointTuple::Tuple{T,T,T}, p::Int=1
) where {T}
    w = [1.0, -2.0, 1.0]
    x = [pointTuple...]
    if p == 1 # Example 3.2 in Bergmann, Laus, Steidl, Weinmann, 2014.
        m = min.(Ref(λ), abs.(x .* w) / (dot(w, w)))
        s = sign.(sum(x .* w))
        return x .- m .* s .* w
    elseif p == 2 # Theorem 3.6 ibd.
        t = λ * sum(x .* w) / (1 + λ * dot(w, w))
        return x .- t .* w
    else
        throw(
            ErrorException(
                "Proximal Map of TV2(Euclidean,λ,pT,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
end
function grad_second_order_Total_Variation(M::NONMUTATINGMANIFOLDS, q, p::Int=1)
    c = mid_point(M, q[1], q[3], q[2]) # nearest mid point of x and z to y
    d = distance(M, q[2], c)
    innerLog = -log(M, c, q[2])
    X = [zero_vector(M, q[i]) for i in 1:3]
    if p == 2
        X[1] = adjoint_differential_shortest_geodesic_startpoint(
            M, q[1], q[3], 1 / 2, innerLog
        )
        X[2] = -log(M, q[2], c)
        X[3] = adjoint_differential_shortest_geodesic_endpoint(
            M, q[1], q[3], 1 / 2, innerLog
        )
    else
        if d > 0 # gradient case (subdifferential contains zero, see above)
            X[1] = adjoint_differential_shortest_geodesic_startpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
            X[2] = -log(M, q[2], c) / (d^(2 - p))
            X[3] = adjoint_differential_shortest_geodesic_endpoint(
                M, q[1], q[3], 1 / 2, innerLog / (d^(2 - p))
            )
        end
    end
    return X
end
function prox_second_order_Total_Variation(
    ::NONMUTATINGMANIFOLDS, λ, pointTuple::Tuple{T,T,T}, p::Int=1
) where {T}
    w = [1.0, -2.0, 1.0]
    x = [pointTuple...]
    if p == 1 # Theorem 3.5 in Bergmann, Laus, Steidl, Weinmann, 2014.
        sr_dot_xw = sym_rem(sum(x .* w))
        m = min(λ, abs(sr_dot_xw) / (dot(w, w)))
        s = sign(sr_dot_xw)
        return sym_rem.(x .- m .* s .* w)
    elseif p == 2 # Theorem 3.6 ibd.
        t = λ * sym_rem(sum(x .* w)) / (1 + λ * dot(w, w))
        return sym_rem.(x - t .* w)
    else
        throw(
            ErrorException(
                "Proximal Map of TV2(Circle,λ,pT,p) not implemented for p=$(p) (requires p=1 or 2)",
            ),
        )
    end
end
function prox_second_order_Total_Variation(
    M::PowerManifold{𝔽,N}, λ, x, p::Int=1
) where {𝔽,N<:NONMUTATINGMANIFOLDS}
    power_size = power_dimensions(M)
    R = CartesianIndices(power_size)
    d = length(size(x))
    minInd = first(R).I
    maxInd = last(R).I
    y = deepcopy(x)
    for k in 1:d # for all directions
        ek = CartesianIndex(ntuple(i -> (i == k) ? 1 : 0, d)) #k th unit vector
        for l in 0:2
            for i in R # iterate over all pixel
                if (i[k] % 3) == l
                    JForward = i.I .+ ek.I #i + e_k
                    JBackward = i.I .- ek.I # i - e_k
                    if all(JForward .<= maxInd) && all(JBackward .>= minInd)
                        (y[M, JBackward...], y[M, i.I...], y[M, JForward...]) = prox_second_order_Total_Variation(
                            M.manifold,
                            λ,
                            (y[M, JBackward...], y[M, i.I...], y[M, JForward...]),
                            p,
                        )
                    end
                end # if mod 3
            end # i in R
        end # for mod 3
    end # directions
    return y
end

@doc raw"""
    project_collaborative_TV(M, λ, x, Ξ[, p=2,q=1])
    project_collaborative_TV!(M, Θ, λ, x, Ξ[, p=2,q=1])

compute the projection onto collaborative Norm unit (or α-) ball, i.e. of the function

```math
F^q(x) = \sum_{i∈\mathcal G}
  \Bigl( \sum_{j∈\mathcal I_i}
    \sum_{k=1}^d \lVert X_{i,j}\rVert_x^p\Bigr)^\frac{q}{p},
```

where ``\mathcal G`` is the set of indices for ``x∈\mathcal M`` and ``\mathcal I_i``
is the set of its forward neighbors.
The computation can also be done in place of `Θ`.

This is adopted from the paper [Duran, Möller, Sbert, Cremers, SIAM J Imag Sci, 2016](@cite DuranMoelleSbertCremers:2016),
see their Example 3 for details.
"""
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p=2.0, q=1.0, α=1.0)
    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        iRep = (1,)
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($(d)) has to be equal to the number of the previous ones ($(s)) but its not.",
                ),
            )
        end
        iRep = (Integer.(ones(d))..., d)
    end
    if q == 1 # Example 3 case 2
        if p == 1
            normΞ = norm.(Ref(N.manifold), x, Ξ)
            return max.(normΞ .- λ, 0.0) ./ ((normΞ .== 0) .+ normΞ) .* Ξ
        end
        if p == 2 # Example 3 case 3
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims=d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner=iRep)
            end
            # if the norm is zero add 1 to avoid division by zero, also then the
            # nominator is already (max(-λ,0) = 0) so it stays zero then
            return max.(norms .- λ, 0.0) ./ ((norms .== 0) .+ norms) .* Ξ
        end
        throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
    elseif q == Inf
        if p == 2
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims=d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner=iRep)
            end
        elseif p == 1
            norms = sum(norm.(Ref(N.manifold), x, Ξ); dims=d + 1)
            if length(iRep) > 1
                norms = repeat(norms; inner=iRep)
            end
        elseif p == Inf
            norms = norm.(Ref(N.manifold), x, Ξ)
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
        return (α .* Ξ) ./ max.(Ref(α), norms)
    end # end q
    return throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Int, q::Float64=1.0, α=1.0)
    return project_collaborative_TV(N, λ, x, Ξ, Float64(p), q, α)
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Float64, q::Int, α=1.0)
    return project_collaborative_TV(N, λ, x, Ξ, p, Float64(q), α)
end
function project_collaborative_TV(N::PowerManifold, λ, x, Ξ, p::Int, q::Int, α=1.0)
    return project_collaborative_TV(N, λ, x, Ξ, Float64(p), Float64(q), α)
end

function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p=2.0, q=1.0, α=1.0)
    pdims = power_dimensions(N)
    if length(pdims) == 1
        d = 1
        s = 1
        iRep = (1,)
    else
        d = pdims[end]
        s = length(pdims) - 1
        if s != d
            throw(
                ErrorException(
                    "the last dimension ($d) has to be equal to the number of the previous ones ($s) but its not.",
                ),
            )
        end
        iRep = (Integer.(ones(d))..., d)
    end
    if q == 1 # Example 3 case 2
        if p == 1
            normΞ = norm.(Ref(N.manifold), x, Ξ)
            Θ .= max.(normΞ .- λ, 0.0) ./ ((normΞ .== 0) .+ normΞ) .* Ξ
            return Θ
        elseif p == 2 # Example 3 case 3
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims=d + 1))
            if length(iRep) > 1
                norms = repeat(norms; inner=iRep)
            end
            # if the norm is zero add 1 to avoid division by zero, also then the
            # nominator is already (max(-λ,0) = 0) so it stays zero then
            Θ .= max.(norms .- λ, 0.0) ./ ((norms .== 0) .+ norms) .* Ξ
            return Θ
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
    elseif q == Inf
        if p == 2
            norms = sqrt.(sum(norm.(Ref(N.manifold), x, Ξ) .^ 2; dims=d + 1))
            (length(iRep) > 1) && (norms = repeat(norms; inner=iRep))
        elseif p == 1
            norms = sum(norm.(Ref(N.manifold), x, Ξ); dims=d + 1)
            (length(iRep) > 1) && (norms = repeat(norms; inner=iRep))
        elseif p == Inf
            norms = norm.(Ref(N.manifold), x, Ξ)
        else
            throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
        end
        Θ .= (α .* Ξ) ./ max.(Ref(α), norms)
        return Θ
    end # end q
    return throw(ErrorException("The case p=$p, q=$q is not yet implemented"))
end
function project_collaborative_TV!(
    N::PowerManifold, Θ, λ, x, Ξ, p::Int, q::Float64=1.0, α=1.0
)
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, Float64(p), q, α)
end
function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p::Float64, q::Int, α=1.0)
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, p, Float64(q), α)
end
function project_collaborative_TV!(N::PowerManifold, Θ, λ, x, Ξ, p::Int, q::Int, α=1.0)
    return project_collaborative_TV!(N, Θ, λ, x, Ξ, Float64(p), Float64(q), α)
end

# == Forward logs ==
# A helper

# (a) The definition

@doc raw"""
    Y = forward_logs(M,x)
    forward_logs!(M, Y, x)

compute the forward logs ``F`` (generalizing forward differences) occurring,
in the power manifold array, the function

```math
F_i(x) = \sum_{j ∈ \mathcal I_i} \log_{x_i} x_j,\quad i  ∈  \mathcal G,
```

where ``\mathcal G`` is the set of indices of the `PowerManifold` manifold `M` and
``\mathcal I_i`` denotes the forward neighbors of ``i``. This can also be done in place of `ξ`.

# Input
* `M` – a `PowerManifold` manifold
* `x` – a point.

# Output
* `Y` – resulting tangent vector in ``T_x\mathcal M`` representing the logs, where
  ``\mathcal N`` is the power manifold with the number of dimensions added to `size(x)`.
  The computation can be done in place of `Y`.
"""
function forward_logs(M::PowerManifold{𝔽,TM,TSize,TPR}, p) where {𝔽,TM,TSize,TPR}
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    sX = size(p)
    maxInd = last(R).I
    if d > 1
        d2 = fill(1, d + 1)
        d2[d + 1] = d
    else
        d2 = 1
    end
    sN = d > 1 ? [power_size..., d] : [power_size...]
    N = PowerManifold(M.manifold, TPR(), sN...)
    xT = repeat(p; inner=d2)
    X = zero_vector(N, xT)
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I
            J = I .+ 1 .* e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neighbour index as Cartesian Index
                X[N, i.I..., k] = log(M.manifold, p[M, i.I...], p[M, j.I...])
            end
        end # directions
    end # i in R
    return X
end
function forward_logs!(M::PowerManifold{𝔽,TM,TSize,TPR}, X, p) where {𝔽,TM,TSize,TPR}
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    sX = size(p)
    maxInd = last(R).I
    if d > 1
        d2 = fill(1, d + 1)
        d2[d + 1] = d
    else
        d2 = 1
    end
    sN = d > 1 ? [power_size..., d] : [power_size...]
    N = PowerManifold(M.manifold, TPR(), sN...)
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I
            J = I .+ 1 .* e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neighbour index as Cartesian Index
                X[N, i.I..., k] = log(M.manifold, p[M, i.I...], p[M, j.I...])
            else
                X[N, i.I..., k] = zero_vector(M.manifold, p[M, i.I...])
            end
        end # directions
    end # i in R
    return X
end

# (c) adjoint differential

@doc raw"""
    Y = adjoint_differential_forward_logs(M, p, X)
    adjoint_differential_forward_logs!(M, Y, p, X)

Compute the adjoint differential of [`forward_logs`](@ref) ``F`` occurring,
in the power manifold array `p`, the differential of the function

``F_i(p) = \sum_{j ∈ \mathcal I_i} \log_{p_i} p_j``

where ``i`` runs over all indices of the `PowerManifold` manifold `M` and ``\mathcal I_i``
denotes the forward neighbors of ``i``
Let ``n`` be the number dimensions of the `PowerManifold` manifold (i.e. `length(size(x)`)).
Then the input tangent vector lies on the manifold ``\mathcal M' = \mathcal M^n``.
The adjoint differential can be computed in place of `Y`.

# Input

* `M`     – a `PowerManifold` manifold
* `p`     – an array of points on a manifold
* `X`     – a tangent vector to from the n-fold power of `p`, where n is the `ndims` of `p`

# Output

`Y` – resulting tangent vector in ``T_p\mathcal M`` representing the adjoint
  differentials of the logs.
"""
function adjoint_differential_forward_logs(
    M::PowerManifold{𝔽,TM,TSize,TPR}, p, X
) where {𝔽,TM,TSize,TPR}
    Y = zero_vector(M, p)
    return adjoint_differential_forward_logs!(M, Y, p, X)
end
function adjoint_differential_forward_logs!(
    M::PowerManifold{𝔽,TM,TSize,TPR}, Y, p, X
) where {𝔽,TM,TSize,TPR}
    power_size = power_dimensions(M)
    d = length(power_size)
    N = PowerManifold(M.manifold, TPR(), power_size..., d)
    R = CartesianIndices(Tuple(power_size))
    maxInd = last(R).I
    # since we add things in Y, make sure we start at zero.
    zero_vector!(M, Y, p)
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = [i.I...] # array of index
            J = I .+ 1 .* (1:d .== k) #i + e_k is j
            if all(J .<= maxInd) # is this neighbor in range?
                j = CartesianIndex{d}(J...) # neighbour index as Cartesian Index
                Y[M, I...] =
                    Y[M, I...] + adjoint_differential_log_basepoint(
                        M.manifold, p[M, I...], p[M, J...], X[N, I..., k]
                    )
                Y[M, J...] =
                    Y[M, J...] + adjoint_differential_log_argument(
                        M.manifold, p[M, J...], p[M, I...], X[N, I..., k]
                    )
            end
        end # directions
    end # i in R
    return Y
end
@doc raw"""
    Y = differential_forward_logs(M, p, X)
    differential_forward_logs!(M, Y, p, X)

compute the differential of [`forward_logs`](@ref) ``F`` on the `PowerManifold` manifold
`M` at `p` and direction `X` , in the power manifold array, the differential of the function

```math
F_i(x) = \sum_{j ∈ \mathcal I_i} \log_{p_i} p_j, \quad i ∈ \mathcal G,
```

where ``\mathcal G`` is the set of indices of the `PowerManifold` manifold `M`
and ``\mathcal I_i`` denotes the forward neighbors of ``i``.

# Input
* `M`     – a `PowerManifold` manifold
* `p`     – a point.
* `X`     – a tangent vector.

# Output
* `Y` – resulting tangent vector in ``T_x\mathcal N`` representing the differentials of the
    logs, where ``\mathcal N`` is the power manifold with the number of dimensions added
    to `size(x)`. The computation can also be done in place.
"""
function differential_forward_logs(M::PowerManifold, p, X)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    d2 = (d > 1) ? ones(Int, d + 1) + (d - 1) * (1:(d + 1) .== d + 1) : 1
    if d > 1
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size..., d)
    else
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size...)
    end
    Y = zero_vector(N, repeat(p; inner=d2))
    return differential_forward_logs!(M, Y, p, X)
end
function differential_forward_logs!(M::PowerManifold, Y, p, X)
    power_size = power_dimensions(M)
    R = CartesianIndices(Tuple(power_size))
    d = length(power_size)
    maxInd = last(R).I
    e_k_vals = [1 * (1:d .== k) for k in 1:d]
    if d > 1
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size..., d)
    else
        N = PowerManifold(M.manifold, NestedPowerRepresentation(), power_size...)
    end
    for i in R # iterate over all pixel
        for k in 1:d # for all direction combinations
            I = i.I # array of index
            J = I .+ e_k_vals[k] #i + e_k is j
            if all(J .<= maxInd)
                # this is neighbor in range,
                # collects two, namely in kth direction since xi appears as base and arg
                Y[N, I..., k] =
                    differential_log_basepoint(
                        M.manifold, p[M, I...], p[M, J...], X[M, I...]
                    ) .+ differential_log_argument(
                        M.manifold, p[M, I...], p[M, J...], X[M, J...]
                    )
            else
                Y[N, I..., k] = zero_vector(M.manifold, p[M, I...])
            end
        end # directions
    end # i in R
    return Y
end
