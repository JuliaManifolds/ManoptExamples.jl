@doc raw"""
    artificialIn_SAR_image([pts=500])
generate an artificial InSAR image, i.e. phase valued data, of size `pts` x
`pts` points.

This data set was introduced for the numerical examples in [BergmannLausSteidlWeinmann:2014:1](@cite).
"""
function artificialIn_SAR_image(pts::Integer)
    # variables
    # rotation of ellipse
    aEll = 35.0
    cosE = cosd(aEll)
    sinE = sind(aEll)
    aStep = 45.0
    cosA = cosd(aStep)
    sinA = sind(aStep)
    # main and minor axis of the ellipse
    axes_inv = [6, 25]
    # values for the hyperboloid
    mid_point = [0.275; 0.275]
    radius = 0.18
    values = [range(-0.5, 0.5; length = pts)...]
    # Steps
    aSteps = 60.0
    cosS = cosd(aSteps)
    sinS = sind(aSteps)
    l = 0.075
    midP = [-0.475, -0.0625] #.125, .55]
    img = zeros(Float64, pts, pts)
    for j in eachindex(values), i in eachindex(values)
        # ellipse
        Xr = cosE * values[i] - sinE * values[j]
        Yr = cosE * values[j] + sinE * values[i]
        v = axes_inv[1] * Xr^2 + axes_inv[2] * Yr^2
        k1 = v <= 1.0 ? 10.0 * pi * Yr : 0.0
        # circle
        Xr = cosA * values[i] - sinA * values[j]
        Yr = cosA * values[j] + sinA * values[i]
        v = ((Xr - mid_point[1])^2 + (Yr - mid_point[2])^2) / radius^2
        k2 = v <= 1.0 ? 4.0 * pi * (1.0 - v) : 0.0
        #
        Xr = cosS * values[i] - sinS * values[j]
        Yr = cosS * values[j] + sinS * values[i]
        k3 = 0.0
        for m in 1:8
            in_range = (abs(Xr + midP[1] + m * l) + abs(Yr + midP[2] + m * l)) ≤ l
            k3 += in_range ? 2 * pi * (m / 8) : 0.0
        end
        img[i, j] = mod(k1 + k2 + k3 - pi, 2 * pi) + pi
    end
    return img
end

@doc raw"""
    artificial_S2_whirl_image([pts::Int=64])

Generate an artificial image of data on the 2 sphere,

# Arguments
* `pts`: (`64`) size of the image in `pts`×`pts` pixel.

This example dataset was used in the numerical example in Section 5.5 of [LausNikolovaPerschSteidl:2017](@cite)

It is based on [`artificial_S2_rotation_image`](@ref) extended by small whirl patches.
"""
function artificial_S2_whirl_image(pts::Int = 64)
    M = Sphere(2)
    img = artificial_S2_rotation_image(pts, (0.5, 0.5))
    # Set WhirlPatches
    sc = pts / 64
    patchSizes = floor.(sc .* [9, 9, 9, 9, 11, 11, 11, 15, 15, 15, 17, 21])
    patchCenters =
        Integer.(
        floor.(
            sc .*
                [[35, 7] [25, 41] [32, 25] [7, 60] [10, 5] [41, 58] [11, 41] [23, 56] [
                38, 45,
            ] [16, 28] [55, 42] [51, 16]],
        ),
    )
    patchSigns = [1, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1]
    for i in 1:length(patchSizes)
        pS = Integer(patchSizes[i])
        pSH = Integer(floor((patchSizes[i] - 1) / 2))
        pC = patchCenters[:, i]
        pC = max.(Ref(1), pC .- pS) .+ pS
        pSgn = patchSigns[i]
        s = pS % 2 == 0 ? 1 : 0
        r = [pC[1] .+ ((-pSH):(pSH + s)), pC[2] .+ ((-pSH):(pSH + s))]
        patch = artificial_S2_whirl_patch(pS)
        if pSgn == -1 # opposite ?
            patch = -patch
        end
        img[r...] = patch
    end
    return img
end

@doc raw"""
    artificial_S2_whirl_patch([pts=5])

create a whirl within the `pts`×`pts` patch of
[Sphere](https://juliamanifolds.github.io/Manifolds.jl/stable/manifolds/sphere.html)(@ref)`(2)`-valued image data.

These patches are used within [`artificial_S2_whirl_image`](@ref).

# Optional Parameters
* `pts`: (`5`) size of the patch. If the number is odd, the center is the north pole.
"""
function artificial_S2_whirl_patch(pts::Int = 5)
    patch = fill([0.0, 0.0, -1.0], pts, pts)
    scaleFactor = sqrt((pts - 1)^2 / 2) * 3 / π
    for i in 1:pts
        for j in 1:pts
            if i != (pts + 1) / 2 || j != (pts + 1) / 2
                α = atan((j - (pts + 1) / 2), (i - (pts + 1) / 2))
                β = sqrt((j - (pts + 1) / 2)^2 + (i - (pts + 1) / 2)^2) / scaleFactor
                patch[i, j] = [
                    sin(α) * sin(π / 2 - β), -cos(α) * sin(π / 2 - β), cos(π / 2 - β),
                ]
            end
        end
    end
    return patch
end

@doc raw"""
    artificial_SPD_image([pts=64, stepsize=1.5])

create an artificial image of symmetric positive definite matrices of size
`pts`×`pts` pixel with a jump of size `stepsize`.

This dataset was used in the numerical example of Section 5.2 of [BacakBergmannSteidlWeinmann:2016](@cite).
"""
function artificial_SPD_image(pts::Int = 64, stepsize = 1.5)
    r = range(0; stop = 1 - 1 / pts, length = pts)
    v1 = abs.(2 * pi .* r .- pi)
    v2 = pi .* r
    v3 = range(0; stop = 3 * (1 - 1 / pts), length = 2 * pts)
    data = fill(Matrix{Float64}(I, 3, 3), pts, pts)
    for row in 1:pts
        for col in 1:pts
            A = [cos(v1[col]) -sin(v1[col]) 0.0; sin(v1[col]) cos(v1[col]) 0.0; 0.0 0.0 1.0]
            B = [1.0 0.0 0.0; 0.0 cos(v2[row]) -sin(v2[row]); 0.0 sin(v2[row]) cos(v2[row])]
            C = [
                cos(v1[mod(col - row, pts) + 1]) 0 -sin(v1[mod(col - row, pts) + 1])
                0.0 1.0 0.0
                sin(v1[mod(col - row, pts) + 1]) 0.0 cos(v1[mod(col - row, pts) + 1])
            ]
            scale = [
                1 + stepsize / 2 * ((row + col) > pts ? 1 : 0)
                1 + v3[row + col] - stepsize * (col > pts / 2 ? 1 : 0)
                4 - v3[row + col] + stepsize * (row > pts / 2 ? 1 : 0)
            ]
            data[row, col] = Matrix(Symmetric(A * B * C * Diagonal(scale) * C' * B' * A'))
        end
    end
    return data
end

@doc raw"""
    artificial_SPD_image2([pts=64, fraction=.66])

create an artificial image of symmetric positive definite matrices of size
`pts`×`pts` pixel with right hand side `fraction` is moved upwards.

This data set was introduced in the numerical examples of Section of [BergmannPerschSteidl:2016](@cite)
"""
function artificial_SPD_image2(pts = 64, fraction = 0.66)
    Zl = 4.0 * Matrix{Float64}(I, 3, 3)
    # create a first matrix
    α = 2.0 * π / 3
    β = π / 3
    B = [1.0 0.0 0.0; 0.0 cos(β) -sin(β); 0.0 sin(β) cos(β)]
    A = [cos(α) -sin(α) 0.0; sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    Zo = Matrix(Symmetric(A * B * Diagonal([2.0, 4.0, 8.0]) * B' * A'))
    # create a second matrix
    α = -4.0 * π / 3
    β = -π / 3
    B = [1.0 0.0 0.0; 0.0 cos(β) -sin(β); 0.0 sin(β) cos(β)]
    A = [cos(α) -sin(α) 0.0; sin(α) cos(α) 0.0; 0.0 0.0 1.0]
    Zt = A * B * Diagonal([8.0 / sqrt(2.0), 8.0, sqrt(2.0)]) * B' * A'
    data = fill(Matrix{Float64}(I, 3, 3), pts, pts)
    M = SymmetricPositiveDefinite(3)
    for row in 1:pts
        for col in 1:pts
            # (a) from Zo a part to Zt
            C = Zo
            if (row > 1) # in X direction
                C = ManifoldsBase.exp_fused(
                    M,
                    C,
                    log(M, C, Zt),
                    (row - 1) / (2 * (pts - 1)) + ((row > fraction * pts) ? 1 / 2 : 0.0),
                )
            end
            if (col > 1) # and then in Y direction
                C = ManifoldsBase.exp_fused(
                    M,
                    C,
                    vector_transport_to(
                        M, Symmetric(Zo), log(M, Zo, Zl), Symmetric(C), ParallelTransport()
                    ),
                    (col - 1.0) / (pts - 1),
                )
            end
            data[row, col] = C
        end
    end
    return data
end

@doc raw"""
    artificial_S2_rotation_image([pts=64, rotations=(.5,.5)])

Create an image with a rotation on each axis as a parametrization.

# Optional Parameters
* `pts`:       (`64`) number of pixels along one dimension
* `rotations`: (`(.5,.5)`) number of total rotations performed on the axes.

This dataset was used in the numerical example of Section 5.1 of [BacakBergmannSteidlWeinmann:2016](@cite).
"""
function artificial_S2_rotation_image(
        pts::Int = 64, rotations::Tuple{Float64, Float64} = (0.5, 0.5)
    )
    M = Sphere(2)
    img = fill(zeros(3), pts, pts)
    north = [1.0, 0.0, 0.0]
    Rxy(a) = [cos(a) -sin(a) 0.0; sin(a) cos(a) 0.0; 0.0 0.0 1.0]
    Rxz(a) = [cos(a) 0.0 -sin(a); 0.0 1.0 0.0; sin(a) 0.0 cos(a)]
    for i in 1:pts
        for j in 1:pts
            x = i / pts * 2π * rotations[1]
            y = j / pts * 2π * rotations[2]
            img[i, j] = Rxy(x + y) * Rxz(x - y) * [0, 0, 1]
        end
    end
    return img
end
