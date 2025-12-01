using Manifolds, Manopt, ManoptExamples, Test
using LinearAlgebra, SparseArrays, OffsetArrays, RecursiveArrayTools

@testset "Test Variational Assembler (Elastic geodesic under force)" begin
    N = 3

    S = Manifolds.Sphere(2)
    power = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S

    mutable struct VariationalSpace
        manifold::AbstractManifold
        degree::Integer
    end

    test_space = VariationalSpace(S, 1)

    start_interval = 0.4
    end_interval = pi - 0.4
    discrete_time = range(; start = start_interval, stop = end_interval, length = N + 2) # equidistant discrete time points

    y0 = [sin(start_interval), 0, cos(start_interval)] # startpoint of geodesic
    yT = [sin(end_interval), 0, cos(end_interval)] # endpoint of geodesic

    y(t) = [sin(t), 0, cos(t)]
    discretized_y = [y(ti) for ti in discrete_time[2:(end - 1)]]

    mutable struct Mapping{F1 <: Function, F2 <: Function, T}
        value::F1
        derivative::F2
        scaling::T
    end

    transport_by_proj(S, p, X, q) = X - q * (q' * X)
    transport_by_proj_prime(S, p, X, dq) = (- dq * p' - p * dq') * X
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

    global test_matrix_rhs = true

    function (ne::NewtonEquation)(M, VB, p)
        n = manifold_dimension(M)
        ne.A .= spzeros(n, n)
        ne.b .= zeros(n)

        Op = OffsetArray([y0, p..., yT], 0:(length(p) + 1))

        ManoptExamples.get_jacobian!(M, Op, evaluate, ne.A, ne.integrand, ne.transport, ne.time_interval; test_space = ne.test_space)
        @test size(ne.A) == (6, 6)
        ManoptExamples.get_right_hand_side!(M, Op, evaluate, ne.b, ne.integrand, ne.time_interval; test_space = ne.test_space)
        @test size(ne.b) == (6,)
        if test_matrix_rhs
            @test Matrix(ne.A) == [2.847607684272587 2.5279563925639 -1.708239045705826 0.0 0.0 0.0; 0.7717619023715545 2.8476076842725875 0.0 -1.4238038421362937 0.0 0.0; -1.708239045705826 0.0 2.8476076842725857 1.7561944901923452 -1.7082390457058247 0.0; 0.0 -1.4238038421362937 6.584674667307058e-33 2.8476076842725857 0.0 -1.4238038421362922; 0.0 0.0 -1.7082390457058247 0.0 2.8476076842725853 2.5279563925639006; 0.0 0.0 0.0 -1.4238038421362922 0.7717619023715555 2.8476076842725853]
            @test ne.b == [1.1642010138654264, 0.0, 1.0753589805471364e-16, 1.1102230246251565e-16, -1.164201013865427, 2.220446049250313e-16]
            global test_matrix_rhs = false
        end
    end

    function solve_in_basis_repr(problem, newtonstate)
        X_base = (problem.newton_equation.A) \ (-problem.newton_equation.b)
        return get_vector(problem.manifold, newtonstate.p, X_base, DefaultOrthogonalBasis())
    end

    NE = NewtonEquation(power, integrand, test_space, transport, discrete_time)

    st_res = vectorbundle_newton(
        power, TangentBundle(power), NE, discretized_y; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
        stopping_criterion = (StopAfterIteration(15) | StopWhenChangeLess(power, 1.0e-12; outer_norm = Inf, inverse_retraction_method = ProjectionInverseRetraction())),
        retraction_method = ProjectionRetraction(),
        return_state = true
    )

    @test get_solver_result(st_res) == [[0.586381584046008, -0.4818143937953777, 0.6511616756407638], [1.0, -3.0171701154889116e-15, 5.346718841666071e-16], [0.5863815840460105, 0.4818143937953751, -0.6511616756407635]]
end


@testset "Test assembly of simplified right hand side" begin
    N = 3

    S = Manifolds.Sphere(2)
    power = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S

    test_space = VariationalSpace(S, 1)

    start_interval = 0.4
    end_interval = pi - 0.4
    discrete_time = range(; start = start_interval, stop = end_interval, length = N + 2) # equidistant discrete time points

    y0 = [sin(start_interval), 0, cos(start_interval)] # startpoint of geodesic
    yT = [sin(end_interval), 0, cos(end_interval)] # endpoint of geodesic

    y(t) = [sin(t), 0, cos(t)]
    discretized_y = [y(ti) for ti in discrete_time[2:(end - 1)]]

    transport_by_proj(S, p, X, q) = X - q * (q' * X)
    transport_by_proj_prime(S, p, X, dq) = (- dq * p' - p * dq') * X
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

    function (ne::NewtonEquation)(M, VB, p)
        n = manifold_dimension(M)
        ne.A .= spzeros(n, n)
        ne.b .= zeros(n)

        Op = OffsetArray([y0, p..., yT], 0:(length(p) + 1))

        ManoptExamples.get_jacobian!(M, Op, evaluate, ne.A, ne.integrand, ne.transport, ne.time_interval; test_space = ne.test_space)
        ManoptExamples.get_right_hand_side!(M, Op, evaluate, ne.b, ne.integrand, ne.time_interval; test_space = ne.test_space)
    end

    global test_rhs_simplified = true
    function (ne::NewtonEquation)(M, VB, p, p_trial)
        n = manifold_dimension(M)
        btrial = zeros(n)

        Op = OffsetArray([y0, p..., yT], 0:(length(p) + 1))
        Optrial = OffsetArray([y0, p_trial..., yT], 0:(length(p_trial) + 1))

        ManoptExamples.get_right_hand_side_simplified!(M, Op, Optrial, evaluate, btrial, ne.integrand, ne.transport, ne.time_interval; test_space = ne.test_space)
        @test size(btrial) == (6,)
        if test_rhs_simplified
            @test btrial == [-0.12876345443232956, -0.09421034004627316, -7.771561172376096e-16, 1.3322676295501878e-15, 0.1287634544323284, 0.09421034004627227]
            global test_rhs_simplified = false
        end
        return btrial
    end

    function solve_in_basis_repr(problem, newtonstate)
        X_base = (problem.newton_equation.A) \ (-problem.newton_equation.b)
        return get_vector(problem.manifold, newtonstate.p, X_base, DefaultOrthogonalBasis())
    end

    NE = NewtonEquation(power, integrand, test_space, transport, discrete_time)

    st_res = vectorbundle_newton(
        power, TangentBundle(power), NE, discretized_y; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
        stopping_criterion = (StopAfterIteration(15) | StopWhenChangeLess(power, 1.0e-12; outer_norm = Inf, inverse_retraction_method = ProjectionInverseRetraction())),
        retraction_method = ProjectionRetraction(),
        stepsize = Manopt.AffineCovariantStepsize(power, θ_des = 0.05),
        return_state = true
    )

    @test get_solver_result(st_res) == [[0.586381584046008, -0.4818143937953779, 0.6511616756407637], [1.0, -4.024168698300908e-15, 7.937293755429942e-16], [0.5863815840460113, 0.48181439379537433, -0.6511616756407632]]
end

@testset "Test Variational Block Assembler (Inextensible rod)" begin
    N = 3

    S = Manifolds.Sphere(2)
    R3 = Manifolds.Euclidean(3)
    powerS = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S
    powerR3 = PowerManifold(R3, NestedPowerRepresentation(), N) # power manifold of R^3
    powerR3_λ = PowerManifold(R3, NestedPowerRepresentation(), N + 1) # power manifold of R^3
    product = ProductManifold(powerR3, powerS, powerR3_λ) # product manifold

    mutable struct VariationalSpace
        manifold::AbstractManifold
        degree::Integer
    end

    test_spaces = ArrayPartition(VariationalSpace(R3, 1), VariationalSpace(S, 1), VariationalSpace(R3, 0))

    ansatz_spaces = ArrayPartition(VariationalSpace(R3, 1), VariationalSpace(S, 1), VariationalSpace(R3, 0))

    start_interval = 0.0
    end_interval = 1.0
    discrete_time = range(; start = start_interval, stop = end_interval, length = N + 2)

    y0 = [0, 0, 0] # startpoint of rod
    y1 = [0.8, 0, 0] # endpoint of rod

    v0 = 1 / norm([1, 0, 2]) * [1, 0, 2] # start direction of rod
    v1 = 1 / norm([1, 0, 0.8]) * [1, 0, 0.8] # end direction of rod

    y(t) = [t * 0.8, 0.1 * t * (1 - t), 0]
    v(t) = [sin(t * pi / 2 + pi / 4), cos(t * pi / 2 + pi / 4), 0]
    λ(t) = [0.1, 0.1, 0.1]

    discretized_y = [y(ti) for ti in discrete_time[2:(end - 1)]]
    discretized_v = [v(ti) for ti in discrete_time[2:(end - 1)]]
    discretized_λ = [λ(ti) for ti in discrete_time[1:(end - 1)]]

    disc_point = ArrayPartition(discretized_y, discretized_v, discretized_λ)

    mutable struct DifferentiableMapping{F1 <: Function, F2 <: Function}
        value::F1
        derivative::F2
    end

    transport_by_proj(S, p, X, q) = X - q * (q' * X)
    transport_by_proj_prime(S, p, X, dq) = (- dq * p' - p * dq') * X
    transport = DifferentiableMapping(transport_by_proj, transport_by_proj_prime)

    Fy_at(Integrand, y, ydot, T, Tdot) = Tdot' * y.x[3] # y component of F
    Fv_at(Integrand, y, ydot, T, Tdot) = ydot.x[2]' * Tdot - T' * y.x[3] # v component of F
    Fλ_at(Integrand, y, ydot, T, Tdot) = (ydot.x[1] - y.x[2])' * T # λ component of F

    F_prime_yλ_at(Integrand, y, ydot, B, Bdot, T, Tdot) = Tdot' * B # derivative of Fy_at w.r.t. λ (others are zero)
    F_prime_vv_at(Integrand, y, ydot, B, Bdot, T, Tdot) = Bdot' * Tdot # derivative of Fv_at w.r.t. v (others are zero)
    F_prime_λv_at(Integrand, y, ydot, B, Bdot, T, Tdot) = -B' * T # derivative of Fλ_at w.r.t. v (others are zero)

    integrand_vv = DifferentiableMapping(Fv_at, F_prime_vv_at)
    integrand_yλ = DifferentiableMapping(Fy_at, F_prime_yλ_at)
    integrand_λv = DifferentiableMapping(Fλ_at, F_prime_λv_at)

    identity_transport(S, p, X, q) = X
    identity_transport_prime(S, p, X, dq) = 0.0 * X
    id_transport = DifferentiableMapping(identity_transport, identity_transport_prime)

    evaluate(y, i, tloc) = ArrayPartition(
        (1.0 - tloc) * y.x[1][i - 1] + tloc * y.x[1][i],
        (1.0 - tloc) * y.x[2][i - 1] + tloc * y.x[2][i],
        y.x[3][i]
    )

    struct NewtonEq{Fy, Fv, Fλ, TS, AS, T, O, NM, Nrhs}
        integrand_y::Fy
        integrand_v::Fv
        integrand_λ::Fλ
        test_spaces::TS
        ansatz_spaces::AS
        vectortransport::T
        discrete_time_interval::O
        A13::NM
        A22::NM
        A32::NM
        A::NM
        b1::Nrhs
        b2::Nrhs
        b3::Nrhs
        b::Nrhs
    end

    function NewtonEq(M, inty, intv, intλ, test_spaces, ansatz_spaces, VT, discrete_time)
        n1 = Int(manifold_dimension(submanifold(M, 1)))
        n2 = Int(manifold_dimension(submanifold(M, 2)))
        n3 = Int(manifold_dimension(submanifold(M, 3)))

        # non-zero blocks of the Newton matrix
        A13 = spzeros(n1, n3)
        A22 = spzeros(n2, n2)
        A32 = spzeros(n3, n2)

        A = spzeros(n1 + n2 + n3, n1 + n2 + n3)

        b1 = zeros(n1)
        b2 = zeros(n2)
        b3 = zeros(n3)
        b = zeros(n1 + n2 + n3)

        return NewtonEq{typeof(inty), typeof(intv), typeof(intλ), typeof(test_spaces), typeof(ansatz_spaces), typeof(VT), typeof(discrete_time), typeof(A13), typeof(b1)}(inty, intv, intλ, test_spaces, ansatz_spaces, VT, discrete_time, A13, A22, A32, A, b1, b2, b3, b)
    end

    global test_matrix_rhs = true

    function (ne::NewtonEq)(M, VB, p)
        n1 = Int(manifold_dimension(submanifold(M, 1)))
        n2 = Int(manifold_dimension(submanifold(M, 2)))
        n3 = Int(manifold_dimension(submanifold(M, 3)))

        ne.A13 .= spzeros(n1, n3)
        ne.A22 .= spzeros(n2, n2)
        ne.A32 .= spzeros(n3, n2)

        ne.b1 .= zeros(n1)
        ne.b2 .= zeros(n2)
        ne.b3 .= zeros(n3)

        Op_y = OffsetArray([y0, p[M, 1]..., y1], 0:(length(p[M, 1]) + 1))
        Op_v = OffsetArray([v0, p[M, 2]..., v1], 0:(length(p[M, 2]) + 1))
        Op_λ = OffsetArray(p[M, 3], 1:length(p[M, 3]))

        Op = ArrayPartition(Op_y, Op_v, Op_λ)

        # assemble (2,2)-block using the connection
        ManoptExamples.get_jacobian_block!(M, Op, evaluate, ne.A22, ne.integrand_v, ne.vectortransport, ne.discrete_time_interval; row_index = 2, column_index = 2, test_space = ne.test_spaces.x[2], ansatz_space = ne.ansatz_spaces.x[2])
        @test size(ne.A22) == (6, 6)
        # assemble (1,3)-block without connection
        ManoptExamples.get_jacobian_block!(M, Op, evaluate, ne.A13, ne.integrand_y, id_transport, ne.discrete_time_interval; row_index = 1, column_index = 3, test_space = ne.test_spaces.x[1], ansatz_space = ne.ansatz_spaces.x[3])
        @test size(ne.A13) == (9, 12)
        # assemble (3,2)-block without connection
        ManoptExamples.get_jacobian_block!(M, Op, evaluate, ne.A32, ne.integrand_λ, id_transport, ne.discrete_time_interval; row_index = 3, column_index = 2, test_space = ne.test_spaces.x[3], ansatz_space = ne.ansatz_spaces.x[2])
        @test size(ne.A32) == (12, 6)


        ManoptExamples.get_right_hand_side_row!(M, Op, evaluate, ne.b1, ne.integrand_y, ne.discrete_time_interval; row_index = 1, test_space = ne.test_spaces.x[1])
        @test size(ne.b1) == (9,)
        ManoptExamples.get_right_hand_side_row!(M, Op, evaluate, ne.b2, ne.integrand_v, ne.discrete_time_interval, ; row_index = 2, test_space = ne.test_spaces.x[2])
        @test size(ne.b2) == (6,)
        ManoptExamples.get_right_hand_side_row!(M, Op, evaluate, ne.b3, ne.integrand_λ, ne.discrete_time_interval, ; row_index = 3, test_space = ne.test_spaces.x[3])
        @test size(ne.b3) == (12,)


        ne.A .= vcat(
            hcat(spzeros(n1, n1), spzeros(n1, n2), ne.A13),
            hcat(spzeros(n2, n1), ne.A22, ne.A32'),
            hcat(ne.A13', ne.A32, spzeros(n3, n3))
        )
        ne.b .= vcat(ne.b1, ne.b2, ne.b3)

        if test_matrix_rhs
            @test Matrix(ne.A22) == [5.380868154339828 0.0 -3.695518130045147 0.0 0.0 0.0; -9.316796552482789e-17 5.380868154339828 0.0 -4.0 0.0 0.0; -3.695518130045147 0.0 7.416036260090294 0.0 -3.695518130045147 0.0; 0.0 -4.0 0.0 7.416036260090294 0.0 -4.0; 0.0 0.0 -3.695518130045147 0.0 6.594762875032289 0.0; 0.0 0.0 0.0 -4.0 4.39376565883322e-17 6.5947628750322895]
            @test Matrix(ne.A13) == [1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0]
            @test Matrix(ne.A32) == [0.04783542904563623 0.0 0.0 0.0 0.0 0.0; -0.11548494156391084 0.0 0.0 0.0 0.0 0.0; 0.0 -0.125 0.0 0.0 0.0 0.0; 0.04783542904563623 0.0 7.654042494670958e-18 0.0 0.0 0.0; -0.11548494156391084 0.0 -0.125 0.0 0.0 0.0; 0.0 -0.125 0.0 -0.125 0.0 0.0; 0.0 0.0 7.654042494670958e-18 0.0 -0.047835429045636216 0.0; 0.0 0.0 -0.125 0.0 -0.11548494156391084 0.0; 0.0 0.0 0.0 -0.125 0.0 -0.125; 0.0 0.0 0.0 0.0 -0.047835429045636216 0.0; 0.0 0.0 0.0 0.0 -0.11548494156391084 0.0; 0.0 0.0 0.0 0.0 0.0 -0.125]
            @test ne.b1 == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            @test ne.b2 == [2.2017687618617314, -3.6027087639996633, -0.02499999999999991, -0.025, -2.758700028480269, -2.523780190217697]
            @test ne.b3 == [0.028613358998594428, -0.029085429045636227, -0.11180339887498948, -0.04048494156391083, -0.04158542904563624, 0.0, -0.040484941563910776, 0.0415854290456362, 0.0, -0.013093542744289671, 0.029085429045636213, -0.07808688094430304]
            global test_matrix_rhs = false
        end

        return
    end

    function solve_in_basis_repr(problem, newtonstate)
        X = (problem.newton_equation.A) \ (-problem.newton_equation.b)
        return get_vector(problem.manifold, newtonstate.p, X, DefaultOrthogonalBasis())
    end

    y_0 = copy(product, disc_point)

    NE = NewtonEq(product, integrand_yλ, integrand_vv, integrand_λv, test_spaces, ansatz_spaces, transport, discrete_time)

    st_res = vectorbundle_newton(
        product, TangentBundle(product), NE, y_0; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
        stopping_criterion = (StopAfterIteration(100) | StopWhenChangeLess(product, 1.0e-12; outer_norm = Inf)),
        retraction_method = ProductRetraction(ExponentialRetraction(), ProjectionRetraction(), ExponentialRetraction()),
        return_state = true
    )

    @test get_solver_result(st_res) == RecursiveArrayTools.ArrayPartition{Float64, Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}, Vector{Vector{Float64}}}}(([[0.17939616371786352, 3.8563733738218966e-42, 0.09246128699172507], [0.3797679482303294, 0.0, -0.02544490371304304], [0.5795183336410239, 2.8698592549372254e-42, -0.10104793173942489]], [[0.9879557142429503, 3.0850986990575173e-41, -0.15473689506611538], [0.615018561856777, -4.591774807899561e-41, -0.7885126305720296], [0.982984521428779, -2.2958874039497803e-41, 0.18368840636097478]], [[-32.46424394115326, -8.609577764811676e-41, 0.7280993274418598], [-32.46424394115326, -8.609577764811676e-41, 0.7280993274418598], [-32.46424394115326, -8.609577764811676e-41, 0.7280993274418598], [-32.46424394115326, -8.609577764811676e-41, 0.7280993274418598]]))
end

@testset "Test assembly simplified right hand side for block system" begin
    N = 3

    S = Manifolds.Sphere(2)
    R3 = Manifolds.Euclidean(3)
    powerS = PowerManifold(S, NestedPowerRepresentation(), N) # power manifold of S
    powerR3 = PowerManifold(R3, NestedPowerRepresentation(), N) # power manifold of R^3
    powerR3_λ = PowerManifold(R3, NestedPowerRepresentation(), N + 1) # power manifold of R^3
    product = ProductManifold(powerR3, powerS, powerR3_λ) # product manifold

    mutable struct VariationalSpace
        manifold::AbstractManifold
        degree::Integer
    end

    test_spaces = ArrayPartition(VariationalSpace(R3, 1), VariationalSpace(S, 1), VariationalSpace(R3, 0))

    ansatz_spaces = ArrayPartition(VariationalSpace(R3, 1), VariationalSpace(S, 1), VariationalSpace(R3, 0))

    start_interval = 0.0
    end_interval = 1.0
    discrete_time = range(; start = start_interval, stop = end_interval, length = N + 2)

    y0 = [0, 0, 0] # startpoint of rod
    y1 = [0.8, 0, 0] # endpoint of rod

    v0 = 1 / norm([1, 0, 2]) * [1, 0, 2] # start direction of rod
    v1 = 1 / norm([1, 0, 0.8]) * [1, 0, 0.8] # end direction of rod

    y(t) = [t * 0.8, 0.1 * t * (1 - t), 0]
    v(t) = [sin(t * pi / 2 + pi / 4), cos(t * pi / 2 + pi / 4), 0]
    λ(t) = [0.1, 0.1, 0.1]

    discretized_y = [y(ti) for ti in discrete_time[2:(end - 1)]]
    discretized_v = [v(ti) for ti in discrete_time[2:(end - 1)]]
    discretized_λ = [λ(ti) for ti in discrete_time[1:(end - 1)]]

    disc_point = ArrayPartition(discretized_y, discretized_v, discretized_λ)

    mutable struct DifferentiableMapping{F1 <: Function, F2 <: Function}
        value::F1
        derivative::F2
    end

    transport_by_proj(S, p, X, q) = X - q * (q' * X)
    transport_by_proj_prime(S, p, X, dq) = (- dq * p' - p * dq') * X
    transport = DifferentiableMapping(transport_by_proj, transport_by_proj_prime)

    Fy_at(Integrand, y, ydot, T, Tdot) = Tdot' * y.x[3] # y component of F
    Fv_at(Integrand, y, ydot, T, Tdot) = ydot.x[2]' * Tdot - T' * y.x[3] # v component of F
    Fλ_at(Integrand, y, ydot, T, Tdot) = (ydot.x[1] - y.x[2])' * T # λ component of F

    F_prime_yλ_at(Integrand, y, ydot, B, Bdot, T, Tdot) = Tdot' * B # derivative of Fy_at w.r.t. λ (others are zero)
    F_prime_vv_at(Integrand, y, ydot, B, Bdot, T, Tdot) = Bdot' * Tdot # derivative of Fv_at w.r.t. v (others are zero)
    F_prime_λv_at(Integrand, y, ydot, B, Bdot, T, Tdot) = -B' * T # derivative of Fλ_at w.r.t. v (others are zero)

    integrand_vv = DifferentiableMapping(Fv_at, F_prime_vv_at)
    integrand_yλ = DifferentiableMapping(Fy_at, F_prime_yλ_at)
    integrand_λv = DifferentiableMapping(Fλ_at, F_prime_λv_at)

    identity_transport(S, p, X, q) = X
    identity_transport_prime(S, p, X, dq) = 0.0 * X
    id_transport = DifferentiableMapping(identity_transport, identity_transport_prime)

    evaluate(y, i, tloc) = ArrayPartition(
        (1.0 - tloc) * y.x[1][i - 1] + tloc * y.x[1][i],
        (1.0 - tloc) * y.x[2][i - 1] + tloc * y.x[2][i],
        y.x[3][i]
    )

    struct NewtonEq{Fy, Fv, Fλ, TS, AS, T, O, NM, Nrhs}
        integrand_y::Fy
        integrand_v::Fv
        integrand_λ::Fλ
        test_spaces::TS
        ansatz_spaces::AS
        vectortransport::T
        discrete_time_interval::O
        A13::NM
        A22::NM
        A32::NM
        A::NM
        b1::Nrhs
        b2::Nrhs
        b3::Nrhs
        b::Nrhs
    end

    function NewtonEq(M, inty, intv, intλ, test_spaces, ansatz_spaces, VT, discrete_time)
        n1 = Int(manifold_dimension(submanifold(M, 1)))
        n2 = Int(manifold_dimension(submanifold(M, 2)))
        n3 = Int(manifold_dimension(submanifold(M, 3)))

        # non-zero blocks of the Newton matrix
        A13 = spzeros(n1, n3)
        A22 = spzeros(n2, n2)
        A32 = spzeros(n3, n2)

        A = spzeros(n1 + n2 + n3, n1 + n2 + n3)

        b1 = zeros(n1)
        b2 = zeros(n2)
        b3 = zeros(n3)
        b = zeros(n1 + n2 + n3)

        return NewtonEq{typeof(inty), typeof(intv), typeof(intλ), typeof(test_spaces), typeof(ansatz_spaces), typeof(VT), typeof(discrete_time), typeof(A13), typeof(b1)}(inty, intv, intλ, test_spaces, ansatz_spaces, VT, discrete_time, A13, A22, A32, A, b1, b2, b3, b)
    end

    function (ne::NewtonEq)(M, VB, p)
        n1 = Int(manifold_dimension(submanifold(M, 1)))
        n2 = Int(manifold_dimension(submanifold(M, 2)))
        n3 = Int(manifold_dimension(submanifold(M, 3)))

        ne.A13 .= spzeros(n1, n3)
        ne.A22 .= spzeros(n2, n2)
        ne.A32 .= spzeros(n3, n2)

        ne.b1 .= zeros(n1)
        ne.b2 .= zeros(n2)
        ne.b3 .= zeros(n3)

        Op_y = OffsetArray([y0, p[M, 1]..., y1], 0:(length(p[M, 1]) + 1))
        Op_v = OffsetArray([v0, p[M, 2]..., v1], 0:(length(p[M, 2]) + 1))
        Op_λ = OffsetArray(p[M, 3], 1:length(p[M, 3]))

        Op = ArrayPartition(Op_y, Op_v, Op_λ)

        # assemble (2,2)-block using the connection
        ManoptExamples.get_jacobian_block!(M, Op, evaluate, ne.A22, ne.integrand_v, ne.vectortransport, ne.discrete_time_interval; row_index = 2, column_index = 2, test_space = ne.test_spaces.x[2], ansatz_space = ne.ansatz_spaces.x[2])
        # assemble (1,3)-block without connection
        ManoptExamples.get_jacobian_block!(M, Op, evaluate, ne.A13, ne.integrand_y, id_transport, ne.discrete_time_interval; row_index = 1, column_index = 3, test_space = ne.test_spaces.x[1], ansatz_space = ne.ansatz_spaces.x[3])
        # assemble (3,2)-block without connection
        ManoptExamples.get_jacobian_block!(M, Op, evaluate, ne.A32, ne.integrand_λ, id_transport, ne.discrete_time_interval; row_index = 3, column_index = 2, test_space = ne.test_spaces.x[3], ansatz_space = ne.ansatz_spaces.x[2])


        ManoptExamples.get_right_hand_side_row!(M, Op, evaluate, ne.b1, ne.integrand_y, ne.discrete_time_interval; row_index = 1, test_space = ne.test_spaces.x[1])
        ManoptExamples.get_right_hand_side_row!(M, Op, evaluate, ne.b2, ne.integrand_v, ne.discrete_time_interval, ; row_index = 2, test_space = ne.test_spaces.x[2])
        ManoptExamples.get_right_hand_side_row!(M, Op, evaluate, ne.b3, ne.integrand_λ, ne.discrete_time_interval, ; row_index = 3, test_space = ne.test_spaces.x[3])


        ne.A .= vcat(
            hcat(spzeros(n1, n1), spzeros(n1, n2), ne.A13),
            hcat(spzeros(n2, n1), ne.A22, ne.A32'),
            hcat(ne.A13', ne.A32, spzeros(n3, n3))
        )
        ne.b .= vcat(ne.b1, ne.b2, ne.b3)
        return
    end

    global test_rhs_simplified = true

    function (ne::NewtonEq)(M, VB, p, p_trial)
        n1 = Int(manifold_dimension(submanifold(M, 1)))
        n2 = Int(manifold_dimension(submanifold(M, 2)))
        n3 = Int(manifold_dimension(submanifold(M, 3)))

        btrial_y = zeros(n1)
        btrial_v = zeros(n2)
        btrial_λ = zeros(n3)

        Op_y = OffsetArray([y0, p[M, 1]..., y1], 0:(length(p[M, 1]) + 1))
        Op_v = OffsetArray([v0, p[M, 2]..., v1], 0:(length(p[M, 2]) + 1))
        Op_λ = OffsetArray(p[M, 3], 1:length(p[M, 3]))
        Op = ArrayPartition(Op_y, Op_v, Op_λ)


        Optrial_y = OffsetArray([y0, p_trial[M, 1]..., y1], 0:(length(p_trial[M, 1]) + 1))
        Optrial_v = OffsetArray([v0, p_trial[M, 2]..., v1], 0:(length(p_trial[M, 2]) + 1))
        Optrial_λ = OffsetArray(p_trial[M, 3], 1:length(p_trial[M, 3]))
        Optrial = ArrayPartition(Optrial_y, Optrial_v, Optrial_λ)

        ManoptExamples.get_right_hand_side_simplified_row!(M, Op, Optrial, evaluate, btrial_y, ne.integrand_y, id_transport, ne.discrete_time_interval; row_index = 1, test_space = ne.test_spaces.x[1])
        @test size(btrial_y) == (9,)
        ManoptExamples.get_right_hand_side_simplified_row!(M, Op, Optrial, evaluate, btrial_v, ne.integrand_v, ne.vectortransport, ne.discrete_time_interval; row_index = 2, test_space = ne.test_spaces.x[2])
        @test size(btrial_v) == (6,)
        ManoptExamples.get_right_hand_side_simplified_row!(M, Op, Optrial, evaluate, btrial_λ, ne.integrand_λ, id_transport, ne.discrete_time_interval; row_index = 3, test_space = ne.test_spaces.x[3])
        @test size(btrial_λ) == (12,)

        if test_rhs_simplified
            @test btrial_y == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            @test btrial_v == [-3.723280013759479, 0.6081203339171481, 0.2693259668949266, 4.94041726735329, 3.2630082906440867, 2.000346215506468]
            @test btrial_λ == [0.005955875845116053, 0.005384487521366693, -0.0005826837870468032, 0.01725769634426519, 0.005085198152608786, -0.005736900950447689, 0.018388113935373665, -0.0063827151309787805, -0.007164080676918993, 0.007086293436224536, -0.006083425762220901, -0.0020098635135180967]

            global test_rhs_simplified = false
        end

        return vcat(btrial_y, btrial_v, btrial_λ)
    end

    function solve_in_basis_repr(problem, newtonstate)
        X = (problem.newton_equation.A) \ (-problem.newton_equation.b)
        return get_vector(problem.manifold, newtonstate.p, X, DefaultOrthogonalBasis())
    end

    y_0 = copy(product, disc_point)

    NE = NewtonEq(product, integrand_yλ, integrand_vv, integrand_λv, test_spaces, ansatz_spaces, transport, discrete_time)

    st_res = vectorbundle_newton(
        product, TangentBundle(product), NE, y_0; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
        stopping_criterion = (StopAfterIteration(100) | StopWhenChangeLess(product, 1.0e-12; outer_norm = Inf)),
        retraction_method = ProductRetraction(ExponentialRetraction(), ProjectionRetraction(), ExponentialRetraction()),
        stepsize = Manopt.AffineCovariantStepsize(product, θ_des = 0.5, outer_norm = Inf),
        return_state = true
    )

    @test get_solver_result(st_res) == RecursiveArrayTools.ArrayPartition{Float64, Tuple{Vector{Vector{Float64}}, Vector{Vector{Float64}}, Vector{Vector{Float64}}}}(([[0.16617021062663, 9.311714124516312e-27, 0.052931882501351245], [0.37241790479586706, 1.7784080245941692e-26, -0.08602082039241878], [0.5853942432977951, 8.462899790563452e-27, -0.12209444392842679]], [[0.8821480895130821, 7.44937129961305e-26, -0.4709721309891059], [0.7678334638408145, -6.76053795722677e-27, -0.6406494921610542], [0.9359772441746095, -6.770319832450762e-26, 0.3520605038729901]], [[-26.389657408789546, -1.4772524854400396e-25, -0.3504761360244674], [-26.389657408789546, -1.4772524854400396e-25, -0.3504761360244674], [-26.389657408789546, -1.4772524854400396e-25, -0.3504761360244674], [-26.389657408789546, -1.4772524854400396e-25, -0.3504761360244674]]))
end
