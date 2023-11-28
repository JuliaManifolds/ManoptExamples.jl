function Manopt.artificial_S2_composite_bezier_curve()
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
    ξ0 = π / (8.0 * sqrt(2.0)) .* [1.0, -1.0, 0.0] # staring direction from d0
    b01 = exp(M, d0, ξ0) # b0+
    ξ1 = π / (4.0 * sqrt(2)) .* [1.0, 0.0, 1.0]
    # b02 or b1- and b11 or b1+ are constructed by this vector with opposing sign
    # to achieve a C1 curve
    b02 = exp(M, d1, ξ1)
    b03 = d1
    # 2
    b10 = d1
    b11 = exp(M, d1, -ξ1) # yields c1 condition
    ξ2 = -π / (4 * sqrt(2)) .* [0.0, 1.0, -1.0]
    b12 = exp(M, d2, ξ2)
    b13 = d2
    # 3
    b20 = d2
    b21 = exp(M, d2, -ξ2)
    ξ3 = π / (8.0 * sqrt(2)) .* [-1.0, 1.0, 0.0]
    b22 = exp(M, d3, ξ3)
    b23 = d3
    # hence the matrix of controlpoints for the curve reads
    return [
        BezierSegment([b00, b01, b02, b03]),
        BezierSegment([b10, b11, b12, b13]),
        BezierSegment([b20, b21, b22, b23]),
    ]
end

function Manopt.artificial_S2_lemniscate(p, t::Float64, a::Float64=π / 2.0)
    M = Sphere(2)
    tP = 2.0 * Float64(p[1] >= 0.0) - 1.0 # Take north or south pole
    base = [0.0, 0.0, tP]
    xc = a * (cos(t) / (sin(t)^2 + 1.0))
    yc = a * (cos(t) * sin(t) / (sin(t)^2 + 1.0))
    tV = vector_transport_to(M, base, [xc, yc, 0.0], p, ParallelTransport())
    return exp(M, p, tV)
end
