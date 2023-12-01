#
# == Bézier Curves and de Casteljau Algorithm ==
#
@doc raw"""
    BezierSegment

A type to capture a Bezier segment. With ``n`` points, a Bézier segment of degree ``n-1``
is stored. On the Euclidean manifold, this yields a polynomial of degree ``n-1``.

This type is mainly used to encapsulate the points within a composite Bezier curve, which
consist of an `AbstractVector` of `BezierSegments` where each of the points might
be a nested array on a `PowerManifold` already.

Not that this can also be used to represent tangent vectors on the control points of a segment.

See also: [`de_Casteljau`](@ref).

# Constructor
    BezierSegment(pts::AbstractVector)

Given an abstract vector of `pts` generate the corresponding Bézier segment.
"""
struct BezierSegment{T<:AbstractVector{S} where {S}}
    pts::T
end
#BezierSegment(pts::T) where {T <: AbstractVector{S} where S} = BezierSegment{T}(pts)
Base.show(io::IO, b::BezierSegment) = print(io, "BezierSegment($(b.pts))")

@doc raw"""
    de_Casteljau(M::AbstractManifold, b::BezierSegment NTuple{N,P}) -> Function

return the [Bézier curve](https://en.wikipedia.org/wiki/Bézier_curve)
``β(⋅;b_0,…,b_n): [0,1] → \mathcal M`` defined by the control points
``b_0,…,b_n∈\mathcal M``, ``n∈\mathbb N``, as a [`BezierSegment`](@ref).
This function implements de Casteljau's algorithm [Casteljau, 1959](@cite deCasteljau:1959), [Casteljau, 1963](@cite deCasteljau:1963) generalized
to manifolds by [Popiel, Noakes, J Approx Theo, 2007](@cite PopielNoakes:2007): Let ``γ_{a,b}(t)`` denote the
shortest geodesic connecting ``a,b∈\mathcal M``. Then the curve is defined by the recursion

```math
\begin{aligned}
    β(t;b_0,b_1) &= \gamma_{b_0,b_1}(t)\\
    β(t;b_0,…,b_n) &= \gamma_{β(t;b_0,…,b_{n-1}), β(t;b_1,…,b_n)}(t),
\end{aligned}
```

and `P` is the type of a point on the `Manifold` `M`.

    de_Casteljau(M::AbstractManifold, B::AbstractVector{<:BezierSegment}) -> Function

Given a vector of Bézier segments, i.e. a vector of control points
``B=\bigl( (b_{0,0},…,b_{n_0,0}),…,(b_{0,m},… b_{n_m,m}) \bigr)``,
where the different segments might be of different degree(s) ``n_0,…,n_m``. The resulting
composite Bézier curve ``c_B:[0,m] → \mathcal M`` consists of ``m`` segments which are
Bézier curves.

````math
c_B(t) :=
    \begin{cases}
        β(t; b_{0,0},…,b_{n_0,0}) & \text{ if } t ∈[0,1]\\
        β(t-i; b_{0,i},…,b_{n_i,i}) & \text{ if }
            t∈(i,i+1], \quad i∈\{1,…,m-1\}.
    \end{cases}
````

````julia
de_Casteljau(M::AbstractManifold, b::BezierSegment, t::Real)
de_Casteljau(M::AbstractManifold, B::AbstractVector{<:BezierSegment}, t::Real)
de_Casteljau(M::AbstractManifold, b::BezierSegment, T::AbstractVector) -> AbstractVector
de_Casteljau(
    M::AbstractManifold,
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector
) -> AbstractVector
````

Evaluate the Bézier curve at time `t` or at times `t` in `T`.
"""
de_Casteljau(M::AbstractManifold, ::Any...)
function de_Casteljau(M::AbstractManifold, b::BezierSegment)
    if length(b.pts) == 2
        return t -> shortest_geodesic(M, b.pts[1], b.pts[2], t)
    else
        return t -> shortest_geodesic(
            M,
            de_Casteljau(M, BezierSegment(b.pts[1:(end - 1)]), t),
            de_Casteljau(M, BezierSegment(b.pts[2:end]), t),
            t,
        )
    end
end
function de_Casteljau(M::AbstractManifold, B::AbstractVector{<:BezierSegment})
    length(B) == 1 && return de_Casteljau(M, B[1])
    return function (t)
        ((0 > t) || (t > length(B))) && throw(
            DomainError(
                "Parameter $(t) outside of domain of the composite Bézier curve [0,$(length(B))].",
            ),
        )
        return de_Casteljau(
            M, B[max(ceil(Int, t), 1)], ceil(Int, t) == 0 ? 0.0 : t - ceil(Int, t) + 1
        )
    end
end
# the direct evaluation can be done iteratively
function de_Casteljau(M::AbstractManifold, b::BezierSegment, t::Real)
    if length(b.pts) == 2
        return shortest_geodesic(M, b.pts[1], b.pts[2], t)
    else
        c = deepcopy(b.pts)
        for l in length(c):-1:2 # casteljau on the tree -> forward with interims storage
            c[1:(l - 1)] .= shortest_geodesic.(Ref(M), c[1:(l - 1)], c[2:l], Ref(t))
        end
    end
    return c[1]
end
function de_Casteljau(M::AbstractManifold, B::AbstractVector{<:BezierSegment}, t::Real)
    ((0 > t) || (t > length(B))) && throw(
        DomainError(
            "Parameter $(t) outside of domain of the composite Bézier curve [0,$(length(B))].",
        ),
    )
    return de_Casteljau(
        M, B[max(ceil(Int, t), 1)], ceil(Int, t) == 0 ? 0.0 : t - ceil(Int, t) + 1
    )
end
de_Casteljau(M::AbstractManifold, b, T::AbstractVector) = de_Casteljau.(Ref(M), Ref(b), T)

@doc raw"""
    get_Bezier_junction_tangent_vectors(M::AbstractManifold, B::AbstractVector{<:BezierSegment})
    get_Bezier_junction_tangent_vectors(M::AbstractManifold, b::BezierSegment)

returns the tangent vectors at start and end points of the composite Bézier curve
pointing from a junction point to the first and last
inner control points for each segment of the composite Bezier curve specified by
the control points `B`, either a vector of segments of controlpoints.
"""
function get_Bezier_junction_tangent_vectors(
    M::AbstractManifold, B::AbstractVector{<:BezierSegment}
)
    return cat(
        [[log(M, b.pts[1], b.pts[2]), log(M, b.pts[end], b.pts[end - 1])] for b in B]...,
        ;
        dims=1,
    )
end
function get_Bezier_junction_tangent_vectors(M::AbstractManifold, b::BezierSegment)
    return get_Bezier_junction_tangent_vectors(M, [b])
end

@doc raw"""
    get_Bezier_junctions(M::AbstractManifold, B::AbstractVector{<:BezierSegment})
    get_Bezier_junctions(M::AbstractManifold, b::BezierSegment)

returns the start and end point(s) of the segments of the composite Bézier curve
specified by the control points `B`. For just one segment `b`, its start and end points
are returned.
"""
function get_Bezier_junctions(
    ::AbstractManifold, B::AbstractVector{<:BezierSegment}, double_inner::Bool=false
)
    return cat(
        [double_inner ? [b.pts[[1, end]]...] : [b.pts[1]] for b in B]...,
        double_inner ? [] : [last(last(B).pts)];
        dims=1,
    )
end
function get_Bezier_junctions(::AbstractManifold, b::BezierSegment, ::Bool=false)
    return b.pts[[1, end]]
end

@doc raw"""
    get_Bezier_inner_points(M::AbstractManifold, B::AbstractVector{<:BezierSegment} )
    get_Bezier_inner_points(M::AbstractManifold, b::BezierSegment)

returns the inner (i.e. despite start and end) points of the segments of the
composite Bézier curve specified by the control points `B`. For a single segment `b`,
its inner points are returned
"""
function get_Bezier_inner_points(M::AbstractManifold, B::AbstractVector{<:BezierSegment})
    return cat([[get_Bezier_inner_points(M, b)...] for b in B]...; dims=1)
end
function get_Bezier_inner_points(::AbstractManifold, b::BezierSegment)
    return b.pts[2:(end - 1)]
end

@doc raw"""
    get_Bezier_points(
        M::AbstractManifold,
        B::AbstractVector{<:BezierSegment},
        reduce::Symbol=:default
    )
    get_Bezier_points(M::AbstractManifold, b::BezierSegment, reduce::Symbol=:default)

returns the control points of the segments of the composite Bézier curve
specified by the control points `B`, either a vector of segments of
controlpoints or a.

This method reduces the points depending on the optional `reduce` symbol

* `:default` – no reduction is performed
* `:continuous` – for a continuous function, the junction points are doubled at
  ``b_{0,i}=b_{n_{i-1},i-1}``, so only ``b_{0,i}`` is in the vector.
* `:differentiable` – for a differentiable function additionally
  ``\log_{b_{0,i}}b_{1,i} = -\log_{b_{n_{i-1},i-1}}b_{n_{i-1}-1,i-1}`` holds.
  hence ``b_{n_{i-1}-1,i-1}`` is omitted.

If only one segment is given, all points of `b` – i.e. `b.pts` is returned.
"""
function get_Bezier_points(
    M::AbstractManifold, B::AbstractVector{<:BezierSegment}, reduce::Symbol=:default
)
    return get_Bezier_points(M, B, Val(reduce))
end
function get_Bezier_points(
    ::AbstractManifold, B::AbstractVector{<:BezierSegment}, ::Val{:default}
)
    return cat([[b.pts...] for b in B]...; dims=1)
end
function get_Bezier_points(
    ::AbstractManifold, B::AbstractVector{<:BezierSegment}, ::Val{:continuous}
)
    return cat([[b.pts[1:(end - 1)]...] for b in B]..., [last(last(B).pts)]; dims=1)
end
function get_Bezier_points(
    ::AbstractManifold, B::AbstractVector{<:BezierSegment}, ::Val{:differentiable}
)
    return cat(
        [first(B).pts[1]], [first(B).pts[2]], [[b.pts[3:end]...] for b in B]..., ; dims=1
    )
end
get_Bezier_points(::AbstractManifold, b::BezierSegment, ::Symbol=:default) = b.pts

@doc raw"""
    get_Bezier_degree(M::AbstractManifold, b::BezierSegment)

return the degree of the Bézier curve represented by the tuple `b` of control points on
the manifold `M`, i.e. the number of points minus 1.
"""
get_Bezier_degree(::AbstractManifold, b::BezierSegment) = length(b.pts) - 1

@doc raw"""
    get_Bezier_degrees(M::AbstractManifold, B::AbstractVector{<:BezierSegment})

return the degrees of the components of a composite Bézier curve represented by tuples
in `B` containing points on the manifold `M`.
"""
function get_Bezier_degrees(M::AbstractManifold, B::AbstractVector{<:BezierSegment})
    return get_Bezier_degree.(Ref(M), B)
end

@doc raw"""
    get_Bezier_segments(M::AbstractManifold, c::AbstractArray{P}, d[, s::Symbol=:default])

returns the array of [`BezierSegment`](@ref)s `B` of a composite Bézier curve reconstructed
from an array `c` of points on the manifold `M` and an array of degrees `d`.

There are a few (reduced) representations that can get extended;
see also [`get_Bezier_points`](@ref). For ease of the following, let ``c=(c_1,…,c_k)``
and ``d=(d_1,…,d_m)``, where ``m`` denotes the number of components the composite Bézier
curve consists of. Then

* `:default` – ``k = m + \sum_{i=1}^m d_i`` since each component requires one point more than
  its degree. The points are then ordered in tuples, i.e.
  ````math
  B = \bigl[ [c_1,…,c_{d_1+1}], (c_{d_1+2},…,c_{d_1+d_2+2}],…, [c_{k-m+1+d_m},…,c_{k}] \bigr]
  ````
* `:continuous` – ``k = 1+ \sum_{i=1}{m} d_i``, since for a continuous curve start and end
  point of successive components are the same, so the very first start point and the end
  points are stored.
  ````math
  B = \bigl[ [c_1,…,c_{d_1+1}], [c_{d_1+1},…,c_{d_1+d_2+1}],…, [c_{k-1+d_m},…,b_{k}) \bigr]
  ````
* `:differentiable` – for a differentiable function additionally to the last explanation, also
  the second point of any segment was not stored except for the first segment.
  Hence ``k = 2 - m + \sum_{i=1}{m} d_i`` and at a junction point ``b_n`` with its given prior
  point ``c_{n-1}``, i.e. this is the last inner point of a segment, the first inner point
  in the next segment the junction is computed as
  ``b = \exp_{c_n}(-\log_{c_n} c_{n-1})`` such that the assumed differentiability holds
"""
function get_Bezier_segments(
    M::AbstractManifold, c::Array{P,1}, d, s::Symbol=:default
) where {P}
    ((length(c) == d[1]) && (length(d) == 1)) && return Tuple(c)
    return get_Bezier_segments(M, c, d, Val(s))
end
function get_Bezier_segments(
    ::AbstractManifold, c::Array{P,1}, d, ::Val{:default}
) where {P}
    endindices = cumsum(d .+ 1)
    startindices = endindices - d
    return [BezierSegment(c[si:ei]) for (si, ei) in zip(startindices, endindices)]
end
function get_Bezier_segments(
    ::AbstractManifold, c::Array{P,1}, d, ::Val{:continuous}
) where {P}
    length(c) != (sum(d) + 1) && error(
        "The number of control points $(length(c)) does not match (for degrees $(d) expected $(sum(d)+1) points.",
    )
    nums = d .+ [(i == length(d)) ? 1 : 0 for i in 1:length(d)]
    endindices = cumsum(nums)
    startindices = cumsum(nums) - nums .+ 1
    return [
        [ # for all append the start of the new also as last
            BezierSegment([c[startindices[i]:endindices[i]]..., c[startindices[i + 1]]]) for
            i in 1:(length(startindices) - 1)
        ]..., # despite for the last
        BezierSegment(c[startindices[end]:endindices[end]]),
    ]
end
function get_Bezier_segments(
    M::AbstractManifold, c::Array{P,1}, d, ::Val{:differentiable}
) where {P}
    length(c) != (sum(d .- 1) + 2) && error(
        "The number of control points $(length(c)) does not match (for degrees $(d) expected $(sum(d.-1)+2) points.",
    )
    nums = d .+ [(i == 1) ? 1 : -1 for i in 1:length(d)]
    endindices = cumsum(nums)
    startindices = cumsum(nums) - nums .+ 1
    return [ # for all append the start of the new also as last
        BezierSegment(c[startindices[1]:endindices[1]]),
        [
            BezierSegment([
                c[endindices[i - 1]],
                exp(
                    M,
                    c[endindices[i - 1]],
                    -log(M, c[endindices[i - 1]], c[endindices[i - 1] - 1]),
                ),
                c[startindices[i]:endindices[i]]...,
            ]) for i in 2:length(startindices)
        ]..., # despite for the last
    ]
end

#
# == Acceleration of a Bézier curve ==
#
# (a) Cost
@doc raw"""
    acceleration_Bezier(
        M::AbstractManifold,
        B::AbstractVector{P},
        degrees::AbstractVector{<:Integer},
        T::AbstractVector{<:AbstractFloat},
    ) where {P}

compute the value of the discrete Acceleration of the composite Bezier curve

```math
\sum_{i=1}^{N-1}\frac{d^2_2 [ B(t_{i-1}), B(t_{i}), B(t_{i+1})]}{\Delta_t^3}
```

where for this formula the `pts` along the curve are equispaced and denoted by
``t_i``, ``i=1,…,N``, and ``d_2`` refers to the second order absolute difference [`costTV2`](@ref)
(squared). Note that the Bézier-curve is given in reduces form as a point on a `PowerManifold`,
together with the `degrees` of the segments and assuming a differentiable curve, the segments
can internally be reconstructed.

This acceleration discretization was introduced in [Bergmann, Gousenbourger, Front. Appl. Math. Stat., 2018](@cite BergmannGousenbourger:2018).

# See also

[`grad_acceleration_Bezier`](@ref), [`L2_acceleration_Bezier`](@ref), [`grad_L2_acceleration_Bezier`](@ref)
"""
function acceleration_Bezier(
    M::AbstractManifold,
    B::AbstractVector{P},
    degrees::AbstractVector{<:Integer},
    T::AbstractVector{<:AbstractFloat},
) where {P}
    Bt = get_Bezier_segments(M, B, degrees, :differentiable)
    p = de_Casteljau(M, Bt, T)
    n = length(T)
    f = p[[1, 3:n..., n]]
    b = p[[1, 1:(n - 2)..., n]]
    d = distance.(Ref(M), p, shortest_geodesic.(Ref(M), f, b, Ref(0.5))) .^ 2
    samplingFactor = 1 / (((max(T...) - min(T...)) / (n - 1))^3)
    return samplingFactor * sum(d)
end

@doc raw"""
    L2_acceleration_Bezier(M,B,pts,λ,d)

compute the value of the discrete Acceleration of the composite Bezier curve
together with a data term, i.e.

````math
\frac{λ}{2}\sum_{i=0}^{N} d_{\mathcal M}(d_i, c_B(i))^2+
\sum_{i=1}^{N-1}\frac{d^2_2 [ B(t_{i-1}), B(t_{i}), B(t_{i+1})]}{\Delta_t^3}
````

where for this formula the `pts` along the curve are equispaced and denoted by
``t_i`` and ``d_2`` refers to the second order absolute difference [`costTV2`](@ref)
(squared), the junction points are denoted by ``p_i``, and to each ``p_i`` corresponds
one data item in the manifold points given in `d`. For details on the acceleration
approximation, see [`acceleration_Bezier`](@ref).
Note that the Bézier-curve is given in reduces form as a point on a `PowerManifold`,
together with the `degrees` of the segments and assuming a differentiable curve, the
segments can internally be reconstructed.


# See also

[`grad_L2_acceleration_Bezier`](@ref), [`acceleration_Bezier`](@ref), [`grad_acceleration_Bezier`](@ref)
"""
function L2_acceleration_Bezier(
    M::AbstractManifold,
    B::AbstractVector{P},
    degrees::AbstractVector{<:Integer},
    T::AbstractVector{<:AbstractFloat},
    λ::AbstractFloat,
    d::AbstractVector{P},
) where {P}
    Bt = get_Bezier_segments(M, B, degrees, :differentiable)
    p = get_Bezier_junctions(M, Bt)
    return acceleration_Bezier(M, B, degrees, T) +
           λ / 2 * sum((distance.(Ref(M), p, d)) .^ 2)
end

# (b) Differential
@doc raw"""
    differential_Bezier_control_points(M::AbstractManifold, b::BezierSegment, t::Float, X::BezierSegment)
    differential_Bezier_control_points!(
        M::AbstractManifold,
        Y,
        b::BezierSegment,
        t,
        X::BezierSegment
    )

evaluate the differential of the Bézier curve with respect to its control points
`b` and tangent vectors `X` given in the tangent spaces of the control points. The result
is the “change” of the curve at `t```∈[0,1]``. The computation can be done in place of `Y`.

See [`de_Casteljau`](@ref) for more details on the curve.
"""
function differential_Bezier_control_points(
    M::AbstractManifold, b::BezierSegment, t, X::BezierSegment
)
    # iterative, because recursively would be too many Casteljau evals
    Y = similar(first(X.pts))
    return differential_Bezier_control_points!(M, Y, b, t, X)
end
function differential_Bezier_control_points!(
    M::AbstractManifold, Y, b::BezierSegment, t, X::BezierSegment
)
    # iterative, because recursively would be too many Casteljau evals
    Z = similar(X.pts)
    c = deepcopy(b.pts)
    for l in length(c):-1:2
        Z[1:(l - 1)] .=
            differential_shortest_geodesic_startpoint.(
                Ref(M), c[1:(l - 1)], c[2:l], Ref(t), X.pts[1:(l - 1)]
            ) .+
            differential_shortest_geodesic_endpoint.(
                Ref(M), c[1:(l - 1)], c[2:l], Ref(t), X.pts[2:l]
            )
        c[1:(l - 1)] = shortest_geodesic.(Ref(M), c[1:(l - 1)], c[2:l], Ref(t))
    end
    return copyto!(M, Y, Z[1])
end
@doc raw"""
    differential_Bezier_control_points(
        M::AbstractManifold,
        b::BezierSegment,
        T::AbstractVector,
        X::BezierSegment,
    )
    differential_Bezier_control_points!(
        M::AbstractManifold,
        Y,
        b::BezierSegment,
        T::AbstractVector,
        X::BezierSegment,
    )

evaluate the differential of the Bézier curve with respect to its control points
`b` and tangent vectors `X` in the tangent spaces of the control points. The result
is the “change” of the curve at the points `T`, elementwise in ``t∈[0,1]``.
The computation can be done in place of `Y`.

See [`de_Casteljau`](@ref) for more details on the curve.
"""
function differential_Bezier_control_points(
    M::AbstractManifold, b::BezierSegment, T::AbstractVector, X::BezierSegment
)
    return differential_Bezier_control_points.(Ref(M), Ref(b), T, Ref(X))
end
function differential_Bezier_control_points!(
    M::AbstractManifold, Y, b::BezierSegment, T::AbstractVector, X::BezierSegment
)
    return differential_Bezier_control_points!.(Ref(M), Y, Ref(b), T, Ref(X))
end
@doc raw"""
    differential_Bezier_control_points(
        M::AbstractManifold,
        B::AbstractVector{<:BezierSegment},
        t,
        X::AbstractVector{<:BezierSegment}
    )
    differential_Bezier_control_points!(
        M::AbstractManifold,
        Y::AbstractVector{<:BezierSegment}
        B::AbstractVector{<:BezierSegment},
        t,
        X::AbstractVector{<:BezierSegment}
    )

evaluate the differential of the composite Bézier curve with respect to its
control points `B` and tangent vectors `Ξ` in the tangent spaces of the control
points. The result is the “change” of the curve at `t```∈[0,N]``, which depends
only on the corresponding segment. Here, ``N`` is the length of `B`.
The computation can be done in place of `Y`.

See [`de_Casteljau`](@ref) for more details on the curve.
"""
function differential_Bezier_control_points(
    M::AbstractManifold,
    B::AbstractVector{<:BezierSegment},
    t,
    X::AbstractVector{<:BezierSegment},
)
    if (0 > t) || (t > length(B))
        return throw(
            DomainError(
                t,
                "The parameter $(t) to evaluate the composite Bézier curve at is outside the interval [0,$(length(B))].",
            ),
        )
    end
    seg = max(ceil(Int, t), 1)
    localT = ceil(Int, t) == 0 ? 0.0 : t - seg + 1
    Y = differential_Bezier_control_points(M, B[seg], localT, X[seg])
    if (Integer(t) == seg) && (seg < length(B)) # boundary case, -> seg-1 is also affecting the boundary differential
        Y .+= differential_Bezier_control_points(M, B[seg + 1], localT - 1, X[seg + 1])
    end
    return Y
end
function differential_Bezier_control_points!(
    M::AbstractManifold,
    Y,
    B::AbstractVector{<:BezierSegment},
    t,
    X::AbstractVector{<:BezierSegment},
)
    if (0 > t) || (t > length(B))
        return throw(
            DomainError(
                t,
                "The parameter $(t) to evaluate the composite Bézier curve at is outside the interval [0,$(length(B))].",
            ),
        )
    end
    seg = max(ceil(Int, t), 1)
    localT = ceil(Int, t) == 0 ? 0.0 : t - seg + 1
    differential_Bezier_control_points!(M, Y, B[seg], localT, X[seg])
    if (Integer(t) == seg) && (seg < length(B)) # boundary case, -> seg-1 is also affecting the boundary differential
        Y .+= differential_Bezier_control_points(M, B[seg + 1], localT - 1, X[seg + 1])
    end
    return Y
end

@doc raw"""
    differential_Bezier_control_points(
        M::AbstractManifold,
        B::AbstractVector{<:BezierSegment},
        T::AbstractVector
        Ξ::AbstractVector{<:BezierSegment}
    )
    differential_Bezier_control_points!(
        M::AbstractManifold,
        Θ::AbstractVector{<:BezierSegment}
        B::AbstractVector{<:BezierSegment},
        T::AbstractVector
        Ξ::AbstractVector{<:BezierSegment}
    )

evaluate the differential of the composite Bézier curve with respect to its
control points `B` and tangent vectors `Ξ` in the tangent spaces of the control
points. The result is the “change” of the curve at the points in `T`, which are elementwise
in ``[0,N]``, and each depending the corresponding segment(s). Here, ``N`` is the
length of `B`. For the mutating variant the result is computed in `Θ`.

See [`de_Casteljau`](@ref) for more details on the curve and [Bergmann, Gousenbourger, Front. Appl. Math. Stat., 2018](@cite BergmannGousenbourger:2018).
"""
function differential_Bezier_control_points(
    M::AbstractManifold,
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector,
    Ξ::AbstractVector{<:BezierSegment},
)
    return differential_Bezier_control_points.(Ref(M), Ref(B), T, Ref(Ξ))
end
function differential_Bezier_control_points!(
    M::AbstractManifold,
    Y,
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector,
    Ξ::AbstractVector{<:BezierSegment},
)
    return differential_Bezier_control_points!.(Ref(M), Y, Ref(B), T, Ref(Ξ))
end

# (c) Adjoint Differentials
@doc raw"""
    adjoint_differential_Bezier_control_points(M::AbstractManifold, b::BezierSegment, t, η)
    adjoint_differential_Bezier_control_points!(
        M::AbstractManifold,
        Y::BezierSegment,
        b::BezierSegment,
        t,
        η,
    )

evaluate the adjoint of the differential of a Bézier curve on the manifold `M`
with respect to its control points `b` based on a point `t```∈[0,1]`` on the
curve and a tangent vector ``η∈T_{β(t)}\mathcal M``.
This can be computed in place of `Y`.

See [`de_Casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_Bezier_control_points(
    M::AbstractManifold, b::BezierSegment, t, η
)
    n = length(b.pts)
    if n == 2
        return BezierSegment([
            adjoint_differential_shortest_geodesic_startpoint(M, b.pts[1], b.pts[2], t, η),
            adjoint_differential_shortest_geodesic_endpoint(M, b.pts[1], b.pts[2], t, η),
        ])
    end
    c = [b.pts, [similar.(b.pts[1:l]) for l in (n - 1):-1:2]...]
    for i in 2:(n - 1) # casteljau on the tree -> forward with interims storage
        c[i] .= shortest_geodesic.(Ref(M), c[i - 1][1:(end - 1)], c[i - 1][2:end], Ref(t))
    end
    Y = [η, [similar(η) for i in 1:(n - 1)]...]
    for i in (n - 1):-1:1 # propagate adjoints -> backward without interims storage
        Y[1:(n - i + 1)] .=
            [ # take previous results and add start&end point effects
                adjoint_differential_shortest_geodesic_startpoint.(
                    Ref(M), c[i][1:(end - 1)], c[i][2:end], Ref(t), Y[1:(n - i)]
                )...,
                zero_vector(M, c[i][end]),
            ] .+ [
                zero_vector(M, c[i][1]),
                adjoint_differential_shortest_geodesic_endpoint.(
                    Ref(M), c[i][1:(end - 1)], c[i][2:end], Ref(t), Y[1:(n - i)]
                )...,
            ]
    end
    return BezierSegment(Y)
end
function adjoint_differential_Bezier_control_points!(
    M::AbstractManifold, Y::BezierSegment, b::BezierSegment, t, η
)
    n = length(b.pts)
    if n == 2
        adjoint_differential_shortest_geodesic_startpoint!(
            M, Y.pts[1], b.pts[1], b.pts[2], t, η
        )
        adjoint_differential_shortest_geodesic_endpoint!(
            M, Y.pts[2], b.pts[1], b.pts[2], t, η
        )
        return Y
    end
    c = [b.pts, [similar.(b.pts[1:l]) for l in (n - 1):-1:2]...]
    for i in 2:(n - 1) # casteljau on the tree -> forward with interims storage
        c[i] .= shortest_geodesic.(Ref(M), c[i - 1][1:(end - 1)], c[i - 1][2:end], Ref(t))
    end
    Y.pts[1] = η
    for i in (n - 1):-1:1 # propagate adjoints -> backward without interims storage
        Y.pts[1:(n - i + 1)] .=
            [ # take previous results and add start&end point effects
                adjoint_differential_shortest_geodesic_startpoint.(
                    Ref(M), c[i][1:(end - 1)], c[i][2:end], Ref(t), Y.pts[1:(n - i)]
                )...,
                zero_vector(M, c[i][end]),
            ] .+ [
                zero_vector(M, c[i][1]),
                adjoint_differential_shortest_geodesic_endpoint.(
                    Ref(M), c[i][1:(end - 1)], c[i][2:end], Ref(t), Y.pts[1:(n - i)]
                )...,
            ]
    end
    return Y
end

@doc raw"""
    adjoint_differential_Bezier_control_points(
        M::AbstractManifold,
        b::BezierSegment,
        t::AbstractVector,
        X::AbstractVector,
    )
    adjoint_differential_Bezier_control_points!(
        M::AbstractManifold,
        Y::BezierSegment,
        b::BezierSegment,
        t::AbstractVector,
        X::AbstractVector,
    )
evaluate the adjoint of the differential of a Bézier curve on the manifold `M`
with respect to its control points `b` based on a points `T```=(t_i)_{i=1}^n`` that
are pointwise in `` t_i∈[0,1]`` on the curve and given corresponding tangential
vectors ``X = (η_i)_{i=1}^n``, ``η_i∈T_{β(t_i)}\mathcal M``
This can be computed in place of `Y`.

See [`de_Casteljau`](@ref) for more details on the curve and [Bergmann, Gousenbourger, Front. Appl. Math. Stat., 2018](@cite BergmannGousenbourger:2018)
"""
function adjoint_differential_Bezier_control_points(
    M::AbstractManifold, b::BezierSegment, t::AbstractVector, X::AbstractVector
)
    effects = [
        bt.pts for bt in adjoint_differential_Bezier_control_points.(Ref(M), Ref(b), t, X)
    ]
    return BezierSegment(sum(effects))
end
function adjoint_differential_Bezier_control_points!(
    M::AbstractManifold,
    Y::BezierSegment,
    b::BezierSegment,
    t::AbstractVector,
    X::AbstractVector,
)
    Z = BezierSegment(similar.(Y.pts))
    fill!.(Y.pts, zero(eltype(first(Y.pts))))
    for i in 1:length(t)
        adjoint_differential_Bezier_control_points!(M, Z, b, t[i], X[i])
        Y.pts .+= Z.pts
    end
    return Y
end

@doc raw"""
    adjoint_differential_Bezier_control_points(
        M::AbstractManifold,
        B::AbstractVector{<:BezierSegment},
        t,
        X
    )
    adjoint_differential_Bezier_control_points!(
        M::AbstractManifold,
        Y::AbstractVector{<:BezierSegment},
        B::AbstractVector{<:BezierSegment},
        t,
        X
    )

evaluate the adjoint of the differential of a composite Bézier curve on the
manifold `M` with respect to its control points `b` based on a points `T```=(t_i)_{i=1}^n``
that are pointwise in ``t_i∈[0,1]`` on the curve and given corresponding tangential
vectors ``X = (η_i)_{i=1}^n``, ``η_i∈T_{β(t_i)}\mathcal M``
This can be computed in place of `Y`.

See [`de_Casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_Bezier_control_points(
    M::AbstractManifold, B::AbstractVector{<:BezierSegment}, t, X
)
    Y = broadcast(b -> BezierSegment(zero_vector.(Ref(M), b.pts)), B) # Double broadcast
    return adjoint_differential_Bezier_control_points!(M, Y, B, t, X)
end
function adjoint_differential_Bezier_control_points!(
    M::AbstractManifold,
    Y::AbstractVector{<:BezierSegment},
    B::AbstractVector{<:BezierSegment},
    t,
    X,
)
    # doubly nested broadcast on the Array(Array) of CPs (note broadcast _and_ .)
    if (0 > t) || (t > length(B))
        error(
            "The parameter ",
            t,
            " to evaluate the composite Bézier curve at is outside the interval [0,",
            length(B),
            "].",
        )
    end
    for y in Y
        fill!.(y.pts, zero(eltype(first(y.pts))))
    end
    seg = max(ceil(Int, t), 1)
    localT = ceil(Int, t) == 0 ? 0.0 : t - seg + 1
    adjoint_differential_Bezier_control_points!(M, Y[seg], B[seg], localT, X)
    return Y
end
@doc raw"""
    adjoint_differential_Bezier_control_points(
        M::AbstractManifold,
        T::AbstractVector,
        X::AbstractVector,
    )
    adjoint_differential_Bezier_control_points!(
        M::AbstractManifold,
        Y::AbstractVector{<:BezierSegment},
        T::AbstractVector,
        X::AbstractVector,
    )

Evaluate the adjoint of the differential with respect to the controlpoints at several times `T`.
This can be computed in place of `Y`.

See [`de_Casteljau`](@ref) for more details on the curve.
"""
function adjoint_differential_Bezier_control_points(
    M::AbstractManifold,
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector,
    X::AbstractVector,
)
    Y = broadcast(b -> BezierSegment(zero_vector.(Ref(M), b.pts)), B) # Double broadcast
    return adjoint_differential_Bezier_control_points!(M, Y, B, T, X)
end
function adjoint_differential_Bezier_control_points!(
    M::AbstractManifold,
    Y::AbstractVector{<:BezierSegment},
    B::AbstractVector{<:BezierSegment},
    T::AbstractVector,
    X::AbstractVector,
)
    Z = [BezierSegment(similar.(y.pts)) for y in Y]
    for j in 1:length(T) # for all times
        adjoint_differential_Bezier_control_points!(M, Z, B, T[j], X[j])
        for i in 1:length(Z)
            Y[i].pts .+= Z[i].pts
        end
    end
    return Y
end

# (d) Gradients
@doc raw"""
    grad_acceleration_Bezier(
        M::AbstractManifold,
        B::AbstractVector,
        degrees::AbstractVector{<:Integer}
        T::AbstractVector
    )

compute the gradient of the discretized acceleration of a (composite) Bézier curve ``c_B(t)``
on the `Manifold` `M` with respect to its control points `B` given as a point on the
`PowerManifold` assuming C1 conditions and known `degrees`. The curve is
evaluated at the points given in `T` (elementwise in ``[0,N]``, where ``N`` is the
number of segments of the Bézier curve). The [`get_Bezier_junctions`](@ref) are fixed for
this gradient (interpolation constraint). For the unconstrained gradient,
see [`grad_L2_acceleration_Bezier`](@ref) and set ``λ=0`` therein. This gradient is computed using
`adjoint_Jacobi_field`s. For details, see [Bergmann, Gousenbourger, Front. Appl. Math. Stat., 2018](@cite BergmannGousenbourger:2018).
See [`de_Casteljau`](@ref) for more details on the curve.

# See also

[`acceleration_Bezier`](@ref),  [`grad_L2_acceleration_Bezier`](@ref), [`L2_acceleration_Bezier`](@ref).
"""
function grad_acceleration_Bezier(
    M::AbstractManifold,
    B::AbstractVector,
    degrees::AbstractVector{<:Integer},
    T::AbstractVector,
)
    gradB = _grad_acceleration_Bezier(M, B, degrees, T)
    Bt = get_Bezier_segments(M, B, degrees, :differentiable)
    for k in 1:length(Bt) # we interpolate so we do not move end points
        zero_vector!(M, gradB[k].pts[end], Bt[k].pts[end])
        zero_vector!(M, gradB[k].pts[1], Bt[k].pts[1])
    end
    zero_vector!(M, gradB[end].pts[end], Bt[end].pts[end])
    return get_Bezier_points(M, gradB, :differentiable)
end
function grad_acceleration_Bezier(M::AbstractManifold, b::BezierSegment, T::AbstractVector)
    gradb = _grad_acceleration_Bezier(M, b.pts, [get_Bezier_degree(M, b)], T)[1]
    zero_vector!(M, gradb.pts[1], b.pts[1])
    zero_vector!(M, gradb.pts[end], b.pts[end])
    return gradb
end

@doc raw"""
    grad_L2_acceleration_Bezier(
        M::AbstractManifold,
        B::AbstractVector{P},
        degrees::AbstractVector{<:Integer},
        T::AbstractVector,
        λ,
        d::AbstractVector{P}
    ) where {P}

compute the gradient of the discretized acceleration of a composite Bézier curve
on the `Manifold` `M` with respect to its control points `B` together with a
data term that relates the junction points `p_i` to the data `d` with a weight
``λ`` compared to the acceleration. The curve is evaluated at the points
given in `pts` (elementwise in ``[0,N]``), where ``N`` is the number of segments of
the Bézier curve. The summands are [`grad_distance`](@ref) for the data term
and [`grad_acceleration_Bezier`](@ref) for the acceleration with interpolation constrains.
Here the [`get_Bezier_junctions`](@ref) are included in the optimization, i.e. setting ``λ=0``
yields the unconstrained acceleration minimization. Note that this is ill-posed, since
any Bézier curve identical to a geodesic is a minimizer.

Note that the Bézier-curve is given in reduces form as a point on a `PowerManifold`,
together with the `degrees` of the segments and assuming a differentiable curve, the segments
can internally be reconstructed.

# See also

[`grad_acceleration_Bezier`](@ref), [`L2_acceleration_Bezier`](@ref), [`acceleration_Bezier`](@ref).
"""
function grad_L2_acceleration_Bezier(
    M::AbstractManifold,
    B::AbstractVector{P},
    degrees::AbstractVector{<:Integer},
    T::AbstractVector,
    λ,
    d::AbstractVector{P},
) where {P}
    gradB = _grad_acceleration_Bezier(M, B, degrees, T)
    Bt = get_Bezier_segments(M, B, degrees, :differentiable)
    # add start and end data grad
    # include data term
    for k in 1:length(Bt)
        gradB[k].pts[1] .+= λ * grad_distance(M, d[k], Bt[k].pts[1])
        if k > 1
            gradB[k - 1].pts[end] .+= λ * grad_distance(M, d[k], Bt[k].pts[1])
        end
    end
    gradB[end].pts[end] .+= λ * grad_distance(M, d[end], Bt[end].pts[end])
    return get_Bezier_points(M, gradB, :differentiable)
end

# common helper for the two acceleration grads
function _grad_acceleration_Bezier(
    M::AbstractManifold,
    B::AbstractVector,
    degrees::AbstractVector{<:Integer},
    T::AbstractVector,
)
    Bt = get_Bezier_segments(M, B, degrees, :differentiable)
    n = length(T)
    m = length(Bt)
    p = de_Casteljau(M, Bt, T)
    center = p
    forward = p[[1, 3:n..., n]]
    backward = p[[1, 1:(n - 2)..., n]]
    mid = mid_point.(Ref(M), backward, forward)
    # where the point of interest appears...
    dt = (max(T...) - min(T...)) / (n - 1)
    inner = -2 / ((dt)^3) .* log.(Ref(M), mid, center)
    asForward =
        adjoint_differential_shortest_geodesic_startpoint.(
            Ref(M), forward, backward, Ref(0.5), inner
        )
    asCenter = -2 / ((dt)^3) * log.(Ref(M), center, mid)
    asBackward =
        adjoint_differential_shortest_geodesic_endpoint.(
            Ref(M), forward, backward, Ref(0.5), inner
        )
    # effect of these to the control points is the preliminary gradient
    grad_B = [
        BezierSegment(a.pts .+ b.pts .+ c.pts) for (a, b, c) in zip(
            adjoint_differential_Bezier_control_points(M, Bt, T[[1, 3:n..., n]], asForward),
            adjoint_differential_Bezier_control_points(M, Bt, T, asCenter),
            adjoint_differential_Bezier_control_points(
                M, Bt, T[[1, 1:(n - 2)..., n]], asBackward
            ),
        )
    ]
    for k in 1:(length(Bt) - 1) # add both effects of left and right segments
        X = grad_B[k + 1].pts[1] + grad_B[k].pts[end]
        grad_B[k].pts[end] .= X
        grad_B[k + 1].pts[1] .= X
    end
    # include c0 & C1 condition
    for k in length(Bt):-1:2
        m = length(Bt[k].pts)
        # updates b-
        X1 =
            grad_B[k - 1].pts[end - 1] .+ adjoint_differential_shortest_geodesic_startpoint(
                M, Bt[k - 1].pts[end - 1], Bt[k].pts[1], 2.0, grad_B[k].pts[2]
            )
        # update b+ - though removed in reduced form
        X2 =
            grad_B[k].pts[2] .+ adjoint_differential_shortest_geodesic_startpoint(
                M, Bt[k].pts[2], Bt[k].pts[1], 2.0, grad_B[k - 1].pts[end - 1]
            )
        # update p - effect from left and right segment as well as from c1 cond
        X3 =
            grad_B[k].pts[1] .+ adjoint_differential_shortest_geodesic_endpoint(
                M, Bt[k - 1].pts[m - 1], Bt[k].pts[1], 2.0, grad_B[k].pts[2]
            )
        # store
        grad_B[k - 1].pts[end - 1] .= X1
        grad_B[k].pts[2] .= X2
        grad_B[k].pts[1] .= X3
        grad_B[k - 1].pts[end] .= X3
    end
    return grad_B
end
