using Manifolds, Test
using ManoptExamples:
    acceleration_Bezier,
    adjoint_differential_Bezier_control_points!,
    artificial_S2_composite_Bezier_curve,
    BezierSegment,
    adjoint_differential_Bezier_control_points,
    differential_Bezier_control_points,
    differential_Bezier_control_points!,
    de_Casteljau,
    get_Bezier_degrees,
    get_Bezier_inner_points,
    get_Bezier_junction_tangent_vectors,
    get_Bezier_junctions,
    get_Bezier_points,
    get_Bezier_segments,
    grad_acceleration_Bezier,
    grad_L2_acceleration_Bezier,
    L2_acceleration_Bezier
using ManifoldDiff:
    adjoint_differential_shortest_geodesic_startpoint,
    adjoint_differential_shortest_geodesic_endpoint,
    grad_distance

@testset "Bezier Tests" begin
    @testset "General Bezier Tests" begin
        repr(BezierSegment([[0.0, 0.0], [0.0, 0.0]])) ==
        "BezierSegment([[0.0, 0.0], [0.0, 0.0]])"
    end
    @testset "Spherical Test" begin
        M = Sphere(2)
        pC = [0.0, 1.0, 0.0]
        pT = exp(M, pC, [0.0, 0.0, 0.7])
        pB = exp(M, pC, [0.0, 0.0, -0.7])
        B = [
            BezierSegment(shortest_geodesic(M, pT, pC, [0.0, 1 / 3, 2 / 3, 1.0])),
            BezierSegment(shortest_geodesic(M, pC, pB, [0.0, 1 / 3, 2 / 3, 1.0])),
        ]
        # this is equispaced, so the pure cost is zero and the gradient is a zero-vector
        t = collect(range(0.0, 1.0; length=5))
        pts = shortest_geodesic(M, pT, pB, t)
        pts2 = de_Casteljau(M, B, 2 .* t)
        @test sum(distance.(Ref(M), pts, pts2)) < 10 * eps()
        aX = log(M, pT, pC)
        aT1 =
            adjoint_differential_Bezier_control_points(
                M, BezierSegment([pT, pC]), 0.5, aX
            ).pts
        aT1a = BezierSegment(similar.(aT1))
        adjoint_differential_Bezier_control_points!(
            M, aT1a, BezierSegment([pT, pC]), 0.5, aX
        )
        @test aT1a.pts == aT1
        aT2 = [
            adjoint_differential_shortest_geodesic_startpoint(M, pT, pC, 0.5, aX),
            adjoint_differential_shortest_geodesic_endpoint(M, pT, pC, 0.5, aX),
        ]
        @test aT1 ≈ aT2
        #
        @test sum(
            norm.(
                grad_acceleration_Bezier(
                    M, B[1], collect(range(0.0, 1.0; length=20))
                ).pts .-
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ),
        ) ≈ 0 atol = 2e-12

        # cost and gradient
        T = collect(range(0.0, 2.0; length=51))
        degrees = get_Bezier_degrees(M, B)
        Bvec = get_Bezier_points(M, B, :differentiable)
        Mp = PowerManifold(M, NestedPowerRepresentation(), length(Bvec))
        @test acceleration_Bezier(M, Bvec, degrees, T) ≈ 0 atol = 10^-10
        z = zero_vector(Mp, Bvec)
        distance(Mp, grad_acceleration_Bezier(M, Bvec, degrees, T), z)
        @test norm(Mp, Bvec, grad_acceleration_Bezier(M, Bvec, degrees, T) - z) ≈ 0 atol =
            2e-12

        d = [pT, exp(M, pC, [0.3, 0.0, 0.0]), pB]
        λ = 3.0

        # cost and gradient with data term
        @test L2_acceleration_Bezier(M, Bvec, degrees, T, λ, [pT, pC, pB]) ≈ 0 atol =
            10^(-10)
        @test L2_acceleration_Bezier(M, Bvec, degrees, T, λ, d) ≈
            λ / 2 * distance(M, d[2], pC) .^ 2
        # when the data are the junctions
        @test norm(
            Mp, Bvec, grad_L2_acceleration_Bezier(M, Bvec, degrees, T, λ, [pT, pC, pB]) - z
        ) ≈ 0 atol = 2e-12
        z[4][1] = -0.9
        @test norm(Mp, Bvec, grad_L2_acceleration_Bezier(M, Bvec, degrees, T, λ, d) - z) ≈ 0 atol =
            2e-12
        # when the data is weighted with zero
        @test L2_acceleration_Bezier(M, Bvec, degrees, T, 0.0, d) ≈ 0 atol = 10^(-10)
        z[4][1] = 0.0
        @test norm(Mp, Bvec, grad_L2_acceleration_Bezier(M, Bvec, degrees, T, 0.0, d) - z) ≈
            0 atol = 2e-12
    end
    @testset "de Casteljau variants" begin
        M = Sphere(2)
        B = artificial_S2_composite_Bezier_curve()
        b = B[2]
        b2s = BezierSegment([b.pts[1], b.pts[end]])
        # (a) 2 points -> geo special case
        f1 = de_Casteljau(M, b2s) # fct -> recursive
        pts1 = f1.([0.0, 0.5, 1.0])
        pts2 = de_Casteljau(M, b2s, [0.0, 0.5, 1.0])
        @test pts1 ≈ pts2
        # (b) one segment
        f2 = de_Casteljau(M, b) # fct -> recursive
        pts3 = f2.([0.0, 0.5, 1.0])
        pts4 = de_Casteljau(M, b, [0.0, 0.5, 1.0])
        @test pts3 ≈ pts4
        # (c) whole composites
        f3 = de_Casteljau(M, B) # fct -> recursive
        pts5 = f3.([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        pts6 = de_Casteljau(M, B, [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        @test pts5 ≈ pts6
        @test_throws DomainError f3(-0.1)
        @test_throws DomainError de_Casteljau(M, B, -0.1)
        @test_throws DomainError f3(3.5)
        @test_throws DomainError de_Casteljau(M, B, 3.5)
    end
    @testset "Spherical Data" begin
        M = Sphere(2)
        B = artificial_S2_composite_Bezier_curve()
        @test de_Casteljau(M, B, [0.0, 1.0, 2.0, 3.0]) ≈
            [B[1].pts[1], B[2].pts[1], B[3].pts[1], B[3].pts[4]]
        @test get_Bezier_junction_tangent_vectors(M, B) ≈ [
            log(M, B[1].pts[1], B[1].pts[2]),
            log(M, B[1].pts[4], B[1].pts[3]),
            log(M, B[2].pts[1], B[2].pts[2]),
            log(M, B[2].pts[4], B[2].pts[3]),
            log(M, B[3].pts[1], B[3].pts[2]),
            log(M, B[3].pts[4], B[3].pts[3]),
        ]
        @test get_Bezier_junction_tangent_vectors(M, B[1]) ≈
            [log(M, B[1].pts[1], B[1].pts[2]), log(M, B[1].pts[4], B[1].pts[3])]
        @test get_Bezier_junctions(M, B[1]) == B[1].pts[[1, end]]
        @test get_Bezier_inner_points(M, B) ==
            [B[1].pts[2], B[1].pts[3], B[2].pts[2], B[2].pts[3], B[3].pts[2], B[3].pts[3]]
        @test get_Bezier_inner_points(M, B[1]) == [B[1].pts[2], B[1].pts[3]]

        @test get_Bezier_points(M, B) == cat([[b.pts...] for b in B]...; dims=1)
        @test get_Bezier_points(M, B, :continuous) ==
            cat([[b.pts[[1:3]...]...] for b in B]..., [B[3].pts[4]]; dims=1)
        @test get_Bezier_points(M, B, :differentiable) ==
            cat([B[1].pts[[1, 2]]...], [b.pts[[3, 4]] for b in B]...; dims=1)
        @test get_Bezier_points(M, B[1]) == B[1].pts
        # for segments just check that they
        d = get_Bezier_degrees(M, B)
        A = get_Bezier_segments(M, get_Bezier_points(M, B), d)
        @test [A[i].pts for i in 1:3] == [B[i].pts for i in 1:3]
        A = get_Bezier_segments(M, get_Bezier_points(M, B, :continuous), d, :continuous)
        @test [A[i].pts for i in 1:3] == [B[i].pts for i in 1:3]
        A = get_Bezier_segments(
            M, get_Bezier_points(M, B, :differentiable), d, :differentiable
        )
        @test [A[i].pts for i in 1:3] == [B[i].pts for i in 1:3]

        # out of range
        @test_throws ErrorException adjoint_differential_Bezier_control_points(
            M, B, 7.0, zero_vector(M, B[1].pts[1])
        )
        # a shortcut to evaluate the adjoint at several points is equal to separate evals
        b = B[2]
        Xi = [log(M, b.pts[1], b.pts[2]), -log(M, b.pts[4], b.pts[3])]
        Xs = adjoint_differential_Bezier_control_points(M, b, [0.0, 1.0], Xi)
        @test isapprox(
            Xs.pts,
            adjoint_differential_Bezier_control_points(
                M, b, 0.0, log(M, b.pts[1], b.pts[2])
            ).pts +
            adjoint_differential_Bezier_control_points(
                M, b, 1.0, -log(M, b.pts[4], b.pts[3])
            ).pts,
        )
        Ys = BezierSegment(similar.(Xs.pts))
        adjoint_differential_Bezier_control_points!(M, Ys, b, [0.0, 1.0], Xi)
        @test isapprox(Xs.pts, Ys.pts)
        # differential
        X = BezierSegment([
            log(M, b.pts[1], b.pts[2]), [zero_vector(M, b.pts[i]) for i in 2:4]...
        ])
        Ye = zero(X.pts[1])
        @test differential_Bezier_control_points(M, b, 0.0, X) ≈ X.pts[1]
        differential_Bezier_control_points!(M, Ye, b, 0.0, X)
        @test Ye ≈ X.pts[1]
        dT1 = differential_Bezier_control_points.(Ref(M), Ref(b), [0.0, 1.0], Ref(X))
        dT2 = differential_Bezier_control_points(M, b, [0.0, 1.0], X)
        dT3 = similar.(dT2)
        differential_Bezier_control_points!(M, dT3, b, [0.0, 1.0], X)
        @test dT1 ≈ dT2
        @test dT3 == dT3
        X2 = [
            BezierSegment([[0.0, 0.0, 0.0] for i in 1:4]),
            X,
            BezierSegment([[0.0, 0.0, 0.0] for i in 1:4]),
        ]
        @test_throws DomainError differential_Bezier_control_points(M, B, 20.0, X2)
        dbT2a = differential_Bezier_control_points(M, B, 1.0, X2)
        dbT3a = similar(dbT2a)
        @test_throws DomainError differential_Bezier_control_points!(M, dbT3a, B, 20.0, X2)
        differential_Bezier_control_points!(M, dbT3a, B, 1.0, X2)
        @test dbT2a == dbT3a
        @test dbT2a ≈ X.pts[1]
        dbT2 = differential_Bezier_control_points(M, B, [1.0, 2.0], X2)
        dbT1 = differential_Bezier_control_points.(Ref(M), Ref(B), [1.0, 2.0], Ref(X2))
        @test dT1 ≈ dbT1
        @test dbT2 ≈ dbT1
        dbT3 = similar.(dbT2)
        differential_Bezier_control_points!(M, dbT3, B, [1.0, 2.0], X2)
        @test dbT2 == dbT3
    end
end
