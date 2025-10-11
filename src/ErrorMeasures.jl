@doc raw"""
    mean_squared_error(M, p, q)

Compute the (mean) squared error between the two
points `p` and `q` on the (power) manifold `M`.
"""
function mean_squared_error(M::mT, p, q) where {mT <: AbstractManifold}
    return distance(M, p, q)^2
end
function mean_squared_error(M::PowerManifold, x, y)
    return 1 / prod(power_dimensions(M)) * sum(distance.(Ref(M.manifold), x, y) .^ 2)
end
@doc raw"""
    mean_average_error(M,x,y)

Compute the (mean) squared error between the two
points `x` and `y` on the `PowerManifold` manifold `M`.
"""
function mean_average_error(M::AbstractManifold, x, y)
    return distance(M, x, y)
end
function mean_average_error(M::PowerManifold, x, y)
    return 1 / prod(power_dimensions(M)) * sum(distance.(Ref(M.manifold), x, y))
end
