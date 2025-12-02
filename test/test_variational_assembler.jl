using Manifolds, Manopt, ManoptExamples, Test
using LinearAlgebra, SparseArrays, OffsetArrays, RecursiveArrayTools

@testset "Variational problem assembler (Elastic geodesic under force)" begin # testset für nicht-Block-Assemblierer bestehend aus drei Teilen
    # gemeinsame Definitionen für alle drei testsets:
    N = 3
    S = Manifolds.Sphere(2)
    mutable struct VariationalSpace
        manifold::AbstractManifold
        degree::Integer
    end

    transport_by_proj(S, p, X, q) = X - q * (q' * X)
    transport_by_proj_prime(S, p, X, dq) = (- dq * p' - p * dq') * X

    power = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S
    test_space = VariationalSpace(S, 1)
    start_interval = 0.4
    end_interval = pi - 0.4
    discrete_time = range(; start = start_interval, stop = end_interval, length = N + 2) # equidistant discrete time points

    γ0 = [sin(start_interval), 0, cos(start_interval)] # startpoint of geodesic
    γT = [sin(end_interval), 0, cos(end_interval)] # endpoint of geodesic

    γ(t) = [sin(t), 0, cos(t)]
    discretized_γ = [γ(ti) for ti in discrete_time[2:(end - 1)]]

    mutable struct Mapping{F1 <: Function, F2 <: Function, T}
        value::F1
        derivative::F2
        scaling::T
    end

    transport = Mapping(transport_by_proj, transport_by_proj_prime, nothing)

    F_at(Integrand, y, ydot, B, Bdot) = ydot' * Bdot + w(y, Integrand.scaling)' * B
    F_prime_at(Integrand, y, ydot, B1, B1dot, B2, B2dot) = B1dot' * B2dot + (w_prime(y, Integrand.scaling) * B1)' * B2
    integrand = Mapping(F_at, F_prime_at, 3.0)

    w(p, c) = c * p[3] * [-p[2] / (p[1]^2 + p[2]^2), p[1] / (p[1]^2 + p[2]^2), 0.0]
    function w_prime(p, c)
        denominator = p[1]^2 + p[2]^2
        return c * [p[3] * 2 * p[1] * p[2] / denominator^2 p[3] * (-1.0 / denominator + 2.0 * p[2]^2 / denominator^2) -p[2] / denominator; p[3] * (1.0 / denominator - 2.0 * p[1]^2 / (denominator^2)) p[3] * (-2.0 * p[1] * p[2] / (denominator^2)) p[1] / denominator; 0.0 0.0 0.0]
    end

    evaluate(p, i, tloc) = (1.0 - tloc) * p[i - 1] + tloc * p[i]

    struct NewtonEquation{F, TS, T, I, NM, Nrhs}
        integrand::F
        test_space::TS
        transport::T
        time_interval::I
        A::NM
        b::Nrhs
    end

    function NewtonEquation(M, F, test_space, VT, interval)
        n = manifold_dimension(M)
        A = spzeros(n, n)
        b = zeros(n)
        return NewtonEquation{typeof(F), typeof(test_space), typeof(VT), typeof(interval), typeof(A), typeof(b)}(F, test_space, VT, interval, A, b)
    end

    function solve_in_basis_repr(problem, newtonstate)
        X_base = (problem.newton_equation.A) \ (-problem.newton_equation.b)
        return get_vector(problem.manifold, newtonstate.p, X_base, DefaultOrthogonalBasis())
    end
    @testset "Test Variational Assembler (Elastic geodesic under force) runs" begin # Testset für Ergebnis der gesamten Newton-Iteration
        function (ne::NewtonEquation)(M, VB, p)
            n = manifold_dimension(M)
            ne.A .= spzeros(n, n)
            ne.b .= zeros(n)

            Op = OffsetArray([γ0, p..., γT], 0:(length(p) + 1))

            ManoptExamples.get_jacobian!(M, Op, evaluate, ne.A, ne.integrand, ne.transport, ne.time_interval; test_space = ne.test_space)
            @test size(ne.A) == (6, 6)
            ManoptExamples.get_right_hand_side!(M, Op, evaluate, ne.b, ne.integrand, ne.time_interval; test_space = ne.test_space)
            @test size(ne.b) == (6,)
        end

        NE = NewtonEquation(power, integrand, test_space, transport, discrete_time)

        st_res = vectorbundle_newton(
            power, TangentBundle(power), NE, discretized_γ; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
            stopping_criterion = (StopAfterIteration(15) | StopWhenChangeLess(power, 1.0e-12; outer_norm = Inf, inverse_retraction_method = ProjectionInverseRetraction())),
            retraction_method = ProjectionRetraction(),
            return_state = true
        )

        @test all(get_solver_result(st_res) .≈ [[0.586381584046008, -0.4818143937953777, 0.6511616756407638], [1.0, -3.0171701154889116e-15, 5.346718841666071e-16], [0.5863815840460105, 0.4818143937953751, -0.6511616756407635]])
    end

    @testset "Test matrix and rhs" begin # Testset für Assemblierung der Jacobi-Matrix und der rechten Seite in der ersten Interation

        function (ne::NewtonEquation)(M, VB, p)
            n = manifold_dimension(M)
            ne.A .= spzeros(n, n)
            ne.b .= zeros(n)

            Op = OffsetArray([γ0, p..., γT], 0:(length(p) + 1))

            ManoptExamples.get_jacobian!(M, Op, evaluate, ne.A, ne.integrand, ne.transport, ne.time_interval; test_space = ne.test_space)
            @test size(ne.A) == (6, 6)
            ManoptExamples.get_right_hand_side!(M, Op, evaluate, ne.b, ne.integrand, ne.time_interval; test_space = ne.test_space)
            @test size(ne.b) == (6,)

            @test all(Matrix(ne.A) .≈ [2.847607684272587 2.5279563925639 -1.708239045705826 0.0 0.0 0.0; 0.7717619023715545 2.8476076842725875 0.0 -1.4238038421362937 0.0 0.0; -1.708239045705826 0.0 2.8476076842725857 1.7561944901923452 -1.7082390457058247 0.0; 0.0 -1.4238038421362937 6.584674667307058e-33 2.8476076842725857 0.0 -1.4238038421362922; 0.0 0.0 -1.7082390457058247 0.0 2.8476076842725853 2.5279563925639006; 0.0 0.0 0.0 -1.4238038421362922 0.7717619023715555 2.8476076842725853])
            @test all(ne.b .≈ [1.1642010138654264, 0.0, 1.0753589805471364e-16, 1.1102230246251565e-16, -1.164201013865427, 2.220446049250313e-16])
        end

        NE = NewtonEquation(power, integrand, test_space, transport, discrete_time)

        st_res2 = vectorbundle_newton(
            power, TangentBundle(power), NE, discretized_γ; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
            stopping_criterion = (StopAfterIteration(1) | StopWhenChangeLess(power, 1.0e-12; outer_norm = Inf, inverse_retraction_method = ProjectionInverseRetraction())),
            retraction_method = ProjectionRetraction(),
            return_state = true
        )
    end

    @testset "Test assembly of simplified right hand side" begin # Testset für Assemblierung der rechte Seite für vereinfachten Newton (für Dämpfung) in der ersten Iteration
        function (ne::NewtonEquation)(M, VB, p, p_trial)
            n = manifold_dimension(M)
            btrial = zeros(n)

            Op = OffsetArray([γ0, p..., γT], 0:(length(p) + 1))
            Optrial = OffsetArray([γ0, p_trial..., γT], 0:(length(p_trial) + 1))

            ManoptExamples.get_right_hand_side_simplified!(M, Op, Optrial, evaluate, btrial, ne.integrand, ne.transport, ne.time_interval; test_space = ne.test_space)
            @test size(btrial) == (6,)
            @test all(btrial .≈ [-0.12876345443232956, -0.09421034004627316, -7.771561172376096e-16, 1.3322676295501878e-15, 0.1287634544323284, 0.09421034004627227])
            return btrial
        end

        NE = NewtonEquation(power, integrand, test_space, transport, discrete_time)

        st_res3 = vectorbundle_newton(
            power, TangentBundle(power), NE, discretized_γ; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
            stopping_criterion = (StopAfterIteration(1) | StopWhenChangeLess(power, 1.0e-12; outer_norm = Inf, inverse_retraction_method = ProjectionInverseRetraction())),
            retraction_method = ProjectionRetraction(),
            stepsize = Manopt.AffineCovariantStepsize(power, θ_des = 0.1),
            return_state = true
        )
    end
end
