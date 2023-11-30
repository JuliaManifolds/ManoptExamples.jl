
function artificial_S2_whirl_image(pts::Int=64)
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
                    38, 45
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

function artificial_SPD_image2(pts=64, fraction=0.66)
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
                C = exp(
                    M,
                    C,
                    log(M, C, Zt),
                    (row - 1) / (2 * (pts - 1)) + ((row > fraction * pts) ? 1 / 2 : 0.0),
                )
            end
            if (col > 1) # and then in Y direction
                C = exp(
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

function artificial_S2_rotation_image(
    pts::Int=64, rotations::Tuple{Float64,Float64}=(0.5, 0.5)
)
    M = Sphere(2)
    img = fill(zeros(3), pts, pts)
    north = [1.0, 0.0, 0.0]
    Rxy(a) = [cos(a) -sin(a) 0.0; sin(a) cos(a) 0.0; 0.0 0.0 1]
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
