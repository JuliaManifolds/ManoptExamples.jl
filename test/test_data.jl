using ManoptExamples, Test, Manifolds

@testset "Data" begin
    @testset "Signals" begin
        s = ManoptExamples.Lemniscate(10)
        @test all(map(p -> is_point(Sphere(2), p), s))
        @test length(s) == 10
        M = Hyperbolic(2)
        p = [0.0, 0.0, 1.0]
        q = ManoptExamples.Lemniscate(0.0; manifold=M, p=p)
        q2 = ManoptExamples.Lemniscate(2π; manifold=M, p=p)
        r = ManoptExamples.Lemniscate(π / 2; manifold=M, p=p)
        isapprox(M; q, q2) # 2π periodic
        isapprox(M; r, p) # 2π periodic
        @test ManoptExamples.artificial_S1_slope_signal(20, 0.0) == repeat([-π / 2], 20)
        @test ismissing(ManoptExamples.artificial_S1_signal(-1.0))
        @test ismissing(ManoptExamples.artificial_S1_signal(2.0))
        @test ManoptExamples.artificial_S1_signal(2) == [-3 * π / 4, -3 * π / 4]
    end

    #=
    @testset "Images" begin
        @test artificialIn_SAR_image(2) == 2 * π * ones(2, 2)
        # for the remainder check data types only
        @test length(artificial_S1_signal(20)) == 20

        @test size(artificial_S2_whirl_image(64)) == (64, 64)
        @test length(artificial_S2_whirl_image(64)[1, 1]) == 3

        @test size(artificial_S2_rotation_image(64)) == (64, 64)
        @test length(artificial_S2_rotation_image(64)[1, 1]) == 3

        @test size(artificial_S2_whirl_patch(8)) == (8, 8)
        @test length(artificial_S2_whirl_patch(8)[1, 1]) == 3

        @test size(artificial_SPD_image(8)) == (8, 8)
        @test size(artificial_SPD_image(8)[1, 1]) == (3, 3)

        @test size(artificial_SPD_image2(8)) == (8, 8)
        @test size(artificial_SPD_image2(8)[1, 1]) == (3, 3)
        @test eltype(artificial_SPD_image2(8)) == Array{Float64,2}
    end
    =#
end
