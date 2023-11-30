using Manifolds
using ManoptExamples:
    differential_forward_logs,
    differential_forward_logs!,
    forward_logs,
    forward_logs!,
    Intrinsic_infimal_convolution_TV12,
    L2_Total_Variation,
    L2_second_order_Total_Variation,
    L2_Total_Variation_1_2,
    grad_intrinsic_infimal_convolution_TV12,
    grad_second_order_Total_Variation,
    grad_second_order_Total_Variation!,
    grad_Total_Variation,
    grad_Total_Variation!,
    project_collaborative_TV,
    project_collaborative_TV!,
    prox_parallel_TV,
    prox_parallel_TV!,
    prox_second_order_Total_Variation,
    prox_second_order_Total_Variation!,
    prox_Total_Variation,
    prox_Total_Variation!,
    second_order_Total_Variation,
    Total_Variation
@testset "Test TV Costs" begin
    M = Sphere(2)
    N = PowerManifold(M, NestedPowerRepresentation(), 3, 3)
    f = repeat([[1.0, 0.0, 0.0]], 3, 3)
    p = repeat([[1.0, 0.0, 0.0]], 3, 3)
    p[2, 1] = [0.0, 1.0, 0.0]

    @test Intrinsic_infimal_convolution_TV12(N, f, f, f, 0.0, 0.0) == 0.0
    @test Intrinsic_infimal_convolution_TV12(N, f, p, p, 0.0, 0.0) ==
        1 / 2 * distance(N, p, f)^2
    @test Intrinsic_infimal_convolution_TV12(N, p, p, p, 2.0, 0.5) ≈
        second_order_Total_Variation(N, p) + Total_Variation(N, p)
    @test Intrinsic_infimal_convolution_TV12(N, p, p, p, 1.0, 0.0) ≈
        second_order_Total_Variation(N, p)
    @test Intrinsic_infimal_convolution_TV12(N, p, p, p, 1.0, 1.0) ≈ Total_Variation(N, p)
    @test L2_second_order_Total_Variation(N, f, 1.0, p) ==
        1 / 2 * distance(N, f, p)^2 + 1.0 * second_order_Total_Variation(N, p)
    #
    @test L2_Total_Variation(N, f, 1.0, f) ≈ 0.0
    @test L2_Total_Variation(N, p, 1.0, p) ≈ 3 * π / 2
    @test L2_Total_Variation_1_2(N, f, 0.0, 1.0, p) ≈
        1 / 2 * distance(N, p, f)^2 + second_order_Total_Variation(N, p)
    @test L2_Total_Variation_1_2(N, f, 1.0, 1.0, p) ≈
        1 / 2 * distance(N, p, f)^2 +
          Total_Variation(N, p) +
          second_order_Total_Variation(N, p)

    @test second_order_Total_Variation(M, Tuple(p[1:3, 1])) ≈ π / 2
    @test Total_Variation(N, p, 1, 2) ≈ sqrt(5 / 4) * π
    @test sum(second_order_Total_Variation(N, p, 1, false)) ==
        second_order_Total_Variation(N, p)
    @test second_order_Total_Variation(N, f, 2) == 0
end

@testset "Test TV proxes" begin
    @testset "proximal maps" begin
        #
        # prox_Total_Variation
        p = [1.0, 0.0, 0.0]
        q = [0.0, 1.0, 0.0]
        M = Sphere(2)
        N = PowerManifold(M, NestedPowerRepresentation(), 2)
        t = similar(p)
        (r, s) = prox_Total_Variation(M, π / 4, (p, q))
        X = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        prox_Total_Variation!(M, X, π / 4, (p, q))
        @test norm(r - s) < eps(Float64)
        @test norm(X[1] - s) < eps(Float64)
        @test norm(X[2] - r) < eps(Float64)
        # i.e. they are moved together
        @test distance(M, r, s) < eps(Float64)
        (t, u) = prox_Total_Variation(M, π / 8, (p, q))
        @test_throws ErrorException prox_Total_Variation(M, π, (p, q), 3)
        @test_throws ErrorException prox_Total_Variation!(M, [p, q], π, (p, q), 3)
        # they cross correlate
        @test (
            abs(t[1] - u[2]) < eps(Float64) &&
            abs(t[2] - u[1]) < eps(Float64) &&
            abs(t[3] - u[3]) < eps(Float64)
        )
        @test distance(M, t, u) ≈ π / 4 # and have moved half their distance
        #
        (v, w) = prox_Total_Variation(M, 1.0, (p, q), 2)
        vC, wC = shortest_geodesic(M, p, q, [1 / 3, 2 / 3])
        @test distance(M, v, vC) ≈ 0
        @test distance(M, w, wC) < eps()
        P = [similar(p), similar(q)]
        prox_Total_Variation!(M, P, 1.0, (p, q), 2)
        @test P == [v, w]
        # prox_Total_Variation on Power
        T = prox_Total_Variation(N, π / 8, [p, q])
        @test distance(N, T, [t, u]) ≈ 0
        # parallelprox_Total_Variation
        N2 = PowerManifold(M, NestedPowerRepresentation(), 3)
        r = geodesic(M, p, q, 0.5)
        s, t = prox_Total_Variation(M, π / 16, (r, q))
        u, v = prox_Total_Variation(M, π / 16, (p, r))
        y = prox_parallel_TV(N2, π / 16, [[p, r, q], [p, r, q]])
        yM = [similar.(ye) for ye in y]
        prox_parallel_TV!(N2, yM, π / 16, [[p, r, q], [p, r, q]])
        @test y == yM
        @test distance(N2, y[1], [p, s, t]) ≈ 0 # even indices in first comp
        @test distance(N2, y[2], [u, v, q]) ≈ 0 # odd in second
        # dimensions of p have to fit, here they don't
        @test_throws ErrorException prox_parallel_TV(N2, π / 16, [[p, r, q]])
        @test_throws ErrorException prox_parallel_TV!(N2, yM, π / 16, [[p, r, q]])
        # prox _second_order_Total_Variation
        p2, r2, q2 = prox_second_order_Total_Variation(M, 1.0, (p, r, q))
        y = [similar(p) for _ in 1:3]
        prox_second_order_Total_Variation!(M, y, 1.0, (p, r, q))
        @test y ≈ [p2, r2, q2]
        sum(distance.(Ref(M), [p, r, q], [p2, r2, q2])) ≈ 0
        @test_throws ErrorException prox_second_order_Total_Variation(M, 1.0, (p, r, q), 2) # since prox_Total_Variation is only defined for p=1
        distance(
            PowerManifold(M, NestedPowerRepresentation(), 3),
            [p2, r2, q2],
            prox_second_order_Total_Variation(
                PowerManifold(M, NestedPowerRepresentation(), 3), 1.0, [p, r, q]
            ),
        ) ≈ 0
        # Circle
        M2 = Circle()
        N2 = PowerManifold(M2, 3)
        pS, rS, qS = [-0.5, 0.1, 0.5]
        d = sum([pS, rS, qS] .* [1.0, -2.0, 1.0])
        m = min(0.3, abs(Manifolds.sym_rem(d) / 6))
        s = sign(Manifolds.sym_rem(d))
        pSc, rSc, qSc = Manifolds.sym_rem.([pS, rS, qS] .- m .* s .* [1.0, -2.0, 1.0])
        pSr, rSr, qSr = prox_second_order_Total_Variation(M2, 0.3, (pS, rS, qS))
        @test sum(distance.(Ref(M2), [pSc, rSc, qSc], [pSr, rSr, qSr])) ≈ 0
        # p=2
        t = 0.3 * Manifolds.sym_rem(d) / (1 + 0.3 * 6.0)
        @test sum(
            distance.(
                Ref(M2),
                [prox_second_order_Total_Variation(M2, 0.3, (pS, rS, qS), 2)...],
                [pS, rS, qS] .- t .* [1.0, -2.0, 1.0],
            ),
        ) ≈ 0
        @test prox_second_order_Total_Variation(N2, 0.3, [pS, rS, qS]) == [pSr, rSr, qSr]
        # others fail
        @test_throws ErrorException prox_second_order_Total_Variation(
            M2, 0.3, (pS, rS, qS), 3
        )
        # Rn
        M3 = Euclidean(1)
        pR, rR, qR = [pS, rS, qS]
        m = min.(Ref(0.3), abs.([pR, rR, qR] .* [1.0, -2.0, 1.0]) / 6)
        s = sign(d)  # we can reuse d
        pRc, rRc, qRc = [pR, rR, qR] .- m .* s .* [1.0, -2.0, 1.0]
        pRr, rRr, qRr = prox_second_order_Total_Variation(M3, 0.3, (pR, rR, qR))
        @test sum(distance.(Ref(M3), [pRc, rRc, qRc], [pRr, rRr, qRr])) ≈ 0
        # p=2
        t = 0.3 * d / (1 + 0.3 * 6.0)
        @test sum(
            distance.(
                Ref(M3),
                [prox_second_order_Total_Variation(M3, 0.3, (pR, rR, qR), 2)...],
                [pR, rR, qR] .- t .* [1.0, -2.0, 1.0],
            ),
        ) ≈ 0
        # others fail
        @test_throws ErrorException prox_second_order_Total_Variation(
            M3, 0.3, (pR, rR, qR), 3
        )
        #
        # collaborative integer tests
        #
        @test_throws ErrorException prox_second_order_Total_Variation(
            M3, 0.3, (pS, rS, qS), 3
        )
        ξR, ηR, νR = [pS, rS, qS]
        N3 = PowerManifold(M3, 3)
        P = [pR rR qR]
        Ξ = [ξR ηR νR]
        Θ = similar(Ξ)
        @test project_collaborative_TV(N3, 0.0, P, Ξ, 1, 1) == Ξ
        project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 1, 1)
        @test Θ == Ξ
        @test project_collaborative_TV(N3, 0.0, P, Ξ, 1.0, 1) == Ξ
        project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 1.0, 1)
        @test Θ == Ξ
        @test project_collaborative_TV(N3, 0.0, P, Ξ, 1, 1.0) == Ξ
        project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 1, 1.0)
        @test Θ == Ξ
        @test project_collaborative_TV(N3, 0.0, P, Ξ, 1.0, 1.0) == Ξ
        project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 1.0, 1.0)
        @test Θ == Ξ

        @test project_collaborative_TV(N3, 0.0, P, Ξ, 2, 1) == Ξ
        project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 2, 1)
        @test Θ == Ξ
        @test norm(N3, P, project_collaborative_TV(N3, 0.0, P, Ξ, 2, Inf)) ≈ norm(Ξ)
        project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 2, Inf)
        @test norm(N3, P, Θ) ≈ norm(Ξ)
        @test sum(abs.(project_collaborative_TV(N3, 0.0, P, Ξ, 1, Inf))) ≈ 1.0
        project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 1, Inf)
        @test sum(abs.(Θ)) ≈ 1.0
        @test norm(N3, P, project_collaborative_TV(N3, 0.0, P, Ξ, Inf, Inf)) ≈ norm(Ξ)
        project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, Inf, Inf)
        @test norm(N3, P, Θ) ≈ norm(Ξ)
        @test_throws ErrorException project_collaborative_TV(N3, 0.0, P, Ξ, 3, 3)
        @test_throws ErrorException project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 3, 3)
        @test_throws ErrorException project_collaborative_TV(N3, 0.0, P, Ξ, 3, 1)
        @test_throws ErrorException project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 3, 1)
        @test_throws ErrorException project_collaborative_TV(N3, 0.0, P, Ξ, 3, Inf)
        @test_throws ErrorException project_collaborative_TV!(N3, Θ, 0.0, P, Ξ, 3, Inf)

        @testset "Multivariate project collaborative TV" begin
            S = Sphere(2)
            M = PowerManifold(S, NestedPowerRepresentation(), 2, 2, 2)
            p = [zeros(3) for i in [1, 2], j in [1, 2], k in [1, 2]]
            p[1, 1, 1] = [1.0, 0.0, 0.0]
            p[1, 2, 1] = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
            p[2, 1, 1] = 1 / sqrt(2) .* [1.0, 0.0, 1.0]
            p[2, 2, 1] = [0.0, 1.0, 0.0]
            p[:, :, 2] = deepcopy(p[:, :, 1])
            X = zero_vector(M, p)
            X[1, 1, 1] .= [0.0, 0.5, 0.5]
            Y = zero_vector(M, p)
            @test norm(project_collaborative_TV(M, 1, p, X, 2, 1)) ≈ 0
            project_collaborative_TV!(M, Y, 1, p, X, 2, 1)
            @test norm(Y) ≈ 0
            @test norm(project_collaborative_TV(M, 0.5, p, X, 2, 1)) ≈
                (norm(X[1, 1, 1]) - 0.5)
            project_collaborative_TV!(M, Y, 0.5, p, X, 2, 1)
            @test norm(Y) ≈ (norm(X[1, 1, 1]) - 0.5)
            Nf = PowerManifold(S, NestedPowerRepresentation(), 2, 2, 1)
            @test_throws ErrorException project_collaborative_TV(Nf, 1, p, X, 2, 1)
            @test_throws ErrorException project_collaborative_TV!(Nf, Y, 1, p, X, 2, 1)
        end
    end
end

@testset "Differentials" begin
    p = [1.0, 0.0, 0.0]
    q = [0.0, 1.0, 0.0]
    M = Sphere(2)
    X = log(M, p, q)
    Y = similar(X)
    @testset "forward logs" begin
        N = PowerManifold(M, NestedPowerRepresentation(), 3)
        x = [p, q, p]
        y = [p, p, q]
        V = [X, zero_vector(M, p), -X]
        Y = Manopt.differential_log_argument(M, p, q, -X)
        W = similar.(V)
        @test norm(
            N,
            x,
            differential_forward_logs(N, x, V) - [-X, [π / 2, 0.0, 0.0], zero_vector(M, p)],
        ) ≈ 0 atol = 8 * 10.0^(-16)
        differential_forward_logs!(N, W, x, V)
        @test norm(N, x, W - [-X, [π / 2, 0.0, 0.0], zero_vector(M, p)]) ≈ 0 atol =
            8 * 10.0^(-16)
        @test isapprox(N, x, Manopt.differential_log_argument(N, x, y, V), [V[1], V[2], Y])
        Manopt.differential_log_argument!(N, W, x, y, V)
        @test isapprox(N, x, W, [V[1], V[2], Y])
    end
    @testset "forward logs on a multivariate power manifold" begin
        S = Sphere(2)
        M = PowerManifold(S, NestedPowerRepresentation(), 2, 2)
        p = [zeros(3) for i in [1, 2], j in [1, 2]]
        p[1, 1] = [1.0, 0.0, 0.0]
        p[1, 2] = 1 / sqrt(2) .* [1.0, 1.0, 0.0]
        p[2, 1] = 1 / sqrt(2) .* [1.0, 0.0, 1.0]
        p[2, 2] = [0.0, 1.0, 0.0]
        t1 = forward_logs(M, p)
        @test t1[1, 1, 1] ≈ log(S, p[1, 1], p[2, 1])
        @test t1[1, 1, 2] ≈ log(S, p[1, 1], p[1, 2])
        @test t1[1, 2, 1] ≈ log(S, p[1, 2], p[2, 2])
        @test t1[1, 2, 2] ≈ log(S, p[1, 2], p[1, 2]) atol = 1e-15
        @test t1[2, 1, 1] ≈ log(S, p[2, 1], p[2, 1]) atol = 1e-15
        @test t1[2, 1, 2] ≈ log(S, p[2, 1], p[2, 2])
        @test t1[2, 2, 1] ≈ log(S, p[2, 2], p[2, 2])
        @test t1[2, 2, 2] ≈ log(S, p[2, 2], p[2, 2])
        t1a = zero.(t1)
        forward_logs!(M, t1a, p)
        @test all(t1 .== t1a)
        X = zero_vector(M, p)
        X[1, 1] .= [0.0, 0.5, 0.5]
        t2 = differential_forward_logs(M, p, X)
        a =
            Manopt.differential_log_basepoint(S, p[1, 1], p[2, 1], X[1, 1]) +
            Manopt.differential_log_argument(S, p[1, 1], p[2, 1], X[2, 1])
        @test t2[1, 1, 1] ≈ a
        @test t2[1, 2, 1] ≈ zero_vector(S, p[1, 2]) atol = 1e-17
        @test t2[2, 1, 1] ≈ zero_vector(S, p[2, 1]) atol = 1e-17
        @test t2[2, 2, 1] ≈ zero_vector(S, p[2, 2]) atol = 1e-17
        b =
            Manopt.differential_log_basepoint(S, p[1, 1], p[1, 2], X[1, 1]) +
            Manopt.differential_log_argument(S, p[1, 1], p[1, 2], X[1, 2])
        @test t2[1, 1, 2] ≈ b
        @test t2[1, 2, 2] ≈ zero_vector(S, p[1, 2]) atol = 1e-17
        @test t2[2, 1, 2] ≈ zero_vector(S, p[2, 1]) atol = 1e-17
        @test t2[2, 2, 2] ≈ zero_vector(S, p[2, 2]) atol = 1e-17
    end
end

@testset "gradients" begin
    @testset "Circle (Allocating)" begin
        M = Circle()
        N = PowerManifold(M, 4)
        x = [0.1, 0.2, 0.3, 0.5]
        tvTestξ = [-1.0, 0.0, 0.0, 1.0]
        @test grad_Total_Variation(N, x) == tvTestξ
        @test grad_Total_Variation(M, (x[1], x[1])) ==
            (zero_vector(M, x[1]), zero_vector(M, x[1]))
        @test norm(N, x, grad_Total_Variation(N, x, 2) - tvTestξ) ≈ 0
        tv2Testξ = [0.0, 1.0, -1.0, 1.0]
        @test grad_second_order_Total_Variation(N, x) == tv2Testξ
        @test norm(N, x, forward_logs(N, x) - [0.1, 0.1, 0.2, 0.0]) ≈ 0 atol = 10^(-16)
        @test norm(
            N,
            x,
            grad_intrinsic_infimal_convolution_TV12(N, x, x, x, 1.0, 1.0)[1] -
            [-1.0, 0.0, 0.0, 1.0],
        ) ≈ 0
        @test norm(N, x, grad_intrinsic_infimal_convolution_TV12(N, x, x, x, 1.0, 1.0)[2]) ≈
            0
        x2 = [0.1, 0.2, 0.3]
        N2 = PowerManifold(M, size(x2)...)
        @test grad_second_order_Total_Variation(N2, x2) == zeros(3)
        @test grad_second_order_Total_Variation(N2, x2, 2) == zeros(3)
        @test grad_Total_Variation(M, (0.0, 0.0), 2) == (0.0, 0.0)
        # 2d forward logs
        N3 = PowerManifold(M, 2, 2)
        N3C = PowerManifold(M, 2, 2, 2)
        x3 = [0.1 0.2; 0.3 0.5]
        x3C = cat(x3, x3; dims=3)
        tC = cat([0.2 0.3; 0.0 0.0], [0.1 0.0; 0.2 0.0]; dims=3)
        @test norm(N3C, x3C, forward_logs(N3, x3) - tC) ≈ 0 atol = 10^(-16)
    end
    @testset "Sphere (Mutating)" begin
        M = Sphere(2)
        p = [0.0, 0.0, 1.0]
        q = [0.0, 1.0, 0.0]
        r = [1.0, 0.0, 0.0]
        @testset "Gradient of total variation" begin
            Y = grad_Total_Variation(M, (p, q))
            Z = [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0]]
            X = similar.(Z)
            grad_Total_Variation!(M, X, (p, q))
            @test [y for y in Y] == X
            @test [y for y in Y] ≈ Z
            N = PowerManifold(M, NestedPowerRepresentation(), 3)
            s = [p, q, r]
            Y2 = grad_Total_Variation(N, s)
            Z2 = [[0.0, -1.0, 0.0], [-1.0, 0.0, -1.0], [0.0, -1.0, 0.0]]
            X2 = zero_vector(N, s)
            grad_Total_Variation!(N, X2, s)
            @test Y2 == Z2
            @test X2 == Z2
            Y2a = grad_Total_Variation(N, s, 2)
            X2a = zero_vector(N, s)
            grad_Total_Variation!(N, X2a, s, 2)
            @test Y2a == X2a
            N2 = PowerManifold(M, NestedPowerRepresentation(), 2)
            Y3 = grad_Total_Variation(M, (p, q), 2)
            X3 = zero_vector(N2, [p, q])
            grad_Total_Variation!(M, X3, (p, q), 2)
            @test [y for y in Y3] == X3
            Y4 = grad_Total_Variation(M, (p, p))
            X4 = zero_vector(N2, [p, q])
            grad_Total_Variation!(M, X4, (p, p))
            @test [y for y in Y4] == X4
        end
        @testset "Grad of second order total variation" begin
            N = PowerManifold(M, NestedPowerRepresentation(), 3)
            s = [p, q, r]
            X = zero_vector(N, s)
            grad_second_order_Total_Variation!(M, X, s)
            Y = grad_second_order_Total_Variation(M, s)
            Z = -1 / sqrt(2) .* [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            @test Y == X
            @test Y ≈ Z
            Y2 = grad_second_order_Total_Variation(M, s, 2)
            Z2 = -1.110720734539 .* [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            @test Y2 ≈ Z2
            s2 = [p, shortest_geodesic(M, p, q, 0.5), q]
            @test grad_second_order_Total_Variation(M, s2) ==
                [zero_vector(M, se) for se in s2]
        end
    end
end
