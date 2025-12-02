using Manifolds, Manopt, ManoptExamples, Test
using LinearAlgebra, SparseArrays, OffsetArrays, RecursiveArrayTools

@testset "Test Variational Block Assembler (Inextensible Rod)" begin # testset for block-assembler (consists of 3 parts)
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

    start_time = 0.0
    end_time = 1.0
    discrete_time_interval = range(; start = start_time, stop = end_time, length = N + 2)

    y0 = [0, 0, 0] # startpoint of rod
    y1 = [0.8, 0, 0] # endpoint of rod

    v0 = 1 / norm([1, 0, 2]) * [1, 0, 2] # start direction of rod
    v1 = 1 / norm([1, 0, 0.8]) * [1, 0, 0.8] # end direction of rod

    y(t) = [t * 0.8, 0.1 * t * (1 - t), 0]
    v(t) = [sin(t * pi / 2 + pi / 4), cos(t * pi / 2 + pi / 4), 0]
    λ(t) = [0.1, 0.1, 0.1]

    discretized_y = [y(ti) for ti in discrete_time_interval[2:(end - 1)]]
    discretized_v = [v(ti) for ti in discrete_time_interval[2:(end - 1)]]
    discretized_λ = [λ(ti) for ti in discrete_time_interval[1:(end - 1)]]

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

    eval(y, i, tloc) = ArrayPartition(
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


    function solve_in_basis_repr(problem, newtonstate)
        X = (problem.newton_equation.A) \ (-problem.newton_equation.b)
        return get_vector(problem.manifold, newtonstate.p, X, DefaultOrthogonalBasis())
    end

    y_0 = copy(product, disc_point)

    @testset "Test Variational Block Assembler (Inextensible rod) runs" begin # testset for result of the whole Newton iteration

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
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A22, ne.integrand_v, ne.vectortransport, ne.discrete_time_interval; row_index = 2, column_index = 2, test_space = ne.test_spaces.x[2], ansatz_space = ne.ansatz_spaces.x[2])
            @test size(ne.A22) == (6, 6)
            # assemble (1,3)-block without connection
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A13, ne.integrand_y, id_transport, ne.discrete_time_interval; row_index = 1, column_index = 3, test_space = ne.test_spaces.x[1], ansatz_space = ne.ansatz_spaces.x[3])
            @test size(ne.A13) == (9, 12)
            # assemble (3,2)-block without connection
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A32, ne.integrand_λ, id_transport, ne.discrete_time_interval; row_index = 3, column_index = 2, test_space = ne.test_spaces.x[3], ansatz_space = ne.ansatz_spaces.x[2])
            @test size(ne.A32) == (12, 6)


            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b1, ne.integrand_y, ne.discrete_time_interval; row_index = 1, test_space = ne.test_spaces.x[1])
            @test size(ne.b1) == (9,)
            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b2, ne.integrand_v, ne.discrete_time_interval, ; row_index = 2, test_space = ne.test_spaces.x[2])
            @test size(ne.b2) == (6,)
            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b3, ne.integrand_λ, ne.discrete_time_interval, ; row_index = 3, test_space = ne.test_spaces.x[3])
            @test size(ne.b3) == (12,)


            ne.A .= vcat(
                hcat(spzeros(n1, n1), spzeros(n1, n2), ne.A13),
                hcat(spzeros(n2, n1), ne.A22, ne.A32'),
                hcat(ne.A13', ne.A32, spzeros(n3, n3))
            )
            ne.b .= vcat(ne.b1, ne.b2, ne.b3)

            return
        end

        NE = NewtonEq(product, integrand_yλ, integrand_vv, integrand_λv, test_spaces, ansatz_spaces, transport, discrete_time_interval)

        st_res = vectorbundle_newton(
            product, TangentBundle(product), NE, y_0; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
            stopping_criterion = (StopAfterIteration(15) | StopWhenChangeLess(product, 1.0e-12; outer_norm = Inf)),
            retraction_method = ProductRetraction(ExponentialRetraction(), ProjectionRetraction(), ExponentialRetraction()),
            return_state = true
        )
        res = get_solver_result(st_res)

        @test norm(res[product, 1] - [[0.17939616371786352, 3.8563733738218966e-42, 0.09246128699172507], [0.3797679482303294, 0.0, -0.02544490371304304], [0.5795183336410239, 2.8698592549372254e-42, -0.10104793173942489]]) + norm(res[product, 1]) ≈ norm(res[product, 1])
        @test norm(res[product, 2] - [[0.9879557142429503, 3.0850986990575173e-41, -0.15473689506611538], [0.615018561856777, -4.591774807899561e-41, -0.7885126305720296], [0.982984521428779, -2.2958874039497803e-41, 0.18368840636097478]]) + norm(res[product, 2]) ≈ norm(res[product, 2])
        @test norm(res[product, 3] - [[-32.46424394115326, -8.609577764811676e-41, 0.7280993274418598], [-32.46424394115326, -8.609577764811676e-41, 0.7280993274418598], [-32.46424394115326, -8.609577764811676e-41, 0.7280993274418598], [-32.46424394115326, -8.609577764811676e-41, 0.7280993274418598]]) + norm(res[product, 3]) ≈ norm(res[product, 3])

    end

    @testset "Test matrix and rhs" begin # testset for assembling the Jacobi matrix and the right-hand side in the first iteration
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
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A22, ne.integrand_v, ne.vectortransport, ne.discrete_time_interval; row_index = 2, column_index = 2, test_space = ne.test_spaces.x[2], ansatz_space = ne.ansatz_spaces.x[2])
            @test size(ne.A22) == (6, 6)
            # assemble (1,3)-block without connection
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A13, ne.integrand_y, id_transport, ne.discrete_time_interval; row_index = 1, column_index = 3, test_space = ne.test_spaces.x[1], ansatz_space = ne.ansatz_spaces.x[3])
            @test size(ne.A13) == (9, 12)
            # assemble (3,2)-block without connection
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A32, ne.integrand_λ, id_transport, ne.discrete_time_interval; row_index = 3, column_index = 2, test_space = ne.test_spaces.x[3], ansatz_space = ne.ansatz_spaces.x[2])
            @test size(ne.A32) == (12, 6)


            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b1, ne.integrand_y, ne.discrete_time_interval; row_index = 1, test_space = ne.test_spaces.x[1])
            @test size(ne.b1) == (9,)
            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b2, ne.integrand_v, ne.discrete_time_interval, ; row_index = 2, test_space = ne.test_spaces.x[2])
            @test size(ne.b2) == (6,)
            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b3, ne.integrand_λ, ne.discrete_time_interval, ; row_index = 3, test_space = ne.test_spaces.x[3])
            @test size(ne.b3) == (12,)


            ne.A .= vcat(
                hcat(spzeros(n1, n1), spzeros(n1, n2), ne.A13),
                hcat(spzeros(n2, n1), ne.A22, ne.A32'),
                hcat(ne.A13', ne.A32, spzeros(n3, n3))
            )
            ne.b .= vcat(ne.b1, ne.b2, ne.b3)

            @test norm(Matrix(ne.A22) - [5.380868154339828 0.0 -3.695518130045147 0.0 0.0 0.0; -9.316796552482789e-17 5.380868154339828 0.0 -4.0 0.0 0.0; -3.695518130045147 0.0 7.416036260090294 0.0 -3.695518130045147 0.0; 0.0 -4.0 0.0 7.416036260090294 0.0 -4.0; 0.0 0.0 -3.695518130045147 0.0 6.594762875032289 0.0; 0.0 0.0 0.0 -4.0 4.39376565883322e-17 6.5947628750322895]) + norm(Matrix(ne.A22)) ≈ norm(Matrix(ne.A22))


            @test norm(Matrix(ne.A13) - [1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 -1.0]) + norm(Matrix(ne.A13)) ≈ norm(Matrix(ne.A13))


            @test norm(Matrix(ne.A32) - [0.04783542904563623 0.0 0.0 0.0 0.0 0.0; -0.11548494156391084 0.0 0.0 0.0 0.0 0.0; 0.0 -0.125 0.0 0.0 0.0 0.0; 0.04783542904563623 0.0 7.654042494670958e-18 0.0 0.0 0.0; -0.11548494156391084 0.0 -0.125 0.0 0.0 0.0; 0.0 -0.125 0.0 -0.125 0.0 0.0; 0.0 0.0 7.654042494670958e-18 0.0 -0.047835429045636216 0.0; 0.0 0.0 -0.125 0.0 -0.11548494156391084 0.0; 0.0 0.0 0.0 -0.125 0.0 -0.125; 0.0 0.0 0.0 0.0 -0.047835429045636216 0.0; 0.0 0.0 0.0 0.0 -0.11548494156391084 0.0; 0.0 0.0 0.0 0.0 0.0 -0.125]) + norm(Matrix(ne.A32)) ≈ norm(Matrix(ne.A32))


            @test norm(ne.b1 - zeros(n1)) + norm(ne.b1) ≈ norm(ne.b1)


            @test norm(ne.b2 - [2.2017687618617314, -3.6027087639996633, -0.02499999999999991, -0.025, -2.758700028480269, -2.523780190217697]) + norm(ne.b2) ≈ norm(ne.b2)


            @test norm(ne.b3 - [0.028613358998594428, -0.029085429045636227, -0.11180339887498948, -0.04048494156391083, -0.04158542904563624, 0.0, -0.040484941563910776, 0.0415854290456362, 0.0, -0.013093542744289671, 0.029085429045636213, -0.07808688094430304]) + norm(ne.b3) ≈ norm(ne.b3)

            return
        end

        NE = NewtonEq(product, integrand_yλ, integrand_vv, integrand_λv, test_spaces, ansatz_spaces, transport, discrete_time_interval)

        st_res = vectorbundle_newton(
            product, TangentBundle(product), NE, y_0; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
            stopping_criterion = (StopAfterIteration(1) | StopWhenChangeLess(product, 1.0e-12; outer_norm = Inf)),
            retraction_method = ProductRetraction(ExponentialRetraction(), ProjectionRetraction(), ExponentialRetraction()),
            return_state = true
        )
    end

    @testset "Test assembly simplified right hand side for block system" begin # testset for assembling the right-hand side for simplified Newton (for damping) in the first iteration
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
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A22, ne.integrand_v, ne.vectortransport, ne.discrete_time_interval; row_index = 2, column_index = 2, test_space = ne.test_spaces.x[2], ansatz_space = ne.ansatz_spaces.x[2])
            @test size(ne.A22) == (6, 6)
            # assemble (1,3)-block without connection
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A13, ne.integrand_y, id_transport, ne.discrete_time_interval; row_index = 1, column_index = 3, test_space = ne.test_spaces.x[1], ansatz_space = ne.ansatz_spaces.x[3])
            @test size(ne.A13) == (9, 12)
            # assemble (3,2)-block without connection
            ManoptExamples.get_jacobian_block!(M, Op, eval, ne.A32, ne.integrand_λ, id_transport, ne.discrete_time_interval; row_index = 3, column_index = 2, test_space = ne.test_spaces.x[3], ansatz_space = ne.ansatz_spaces.x[2])
            @test size(ne.A32) == (12, 6)


            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b1, ne.integrand_y, ne.discrete_time_interval; row_index = 1, test_space = ne.test_spaces.x[1])
            @test size(ne.b1) == (9,)
            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b2, ne.integrand_v, ne.discrete_time_interval, ; row_index = 2, test_space = ne.test_spaces.x[2])
            @test size(ne.b2) == (6,)
            ManoptExamples.get_right_hand_side_row!(M, Op, eval, ne.b3, ne.integrand_λ, ne.discrete_time_interval, ; row_index = 3, test_space = ne.test_spaces.x[3])
            @test size(ne.b3) == (12,)


            ne.A .= vcat(
                hcat(spzeros(n1, n1), spzeros(n1, n2), ne.A13),
                hcat(spzeros(n2, n1), ne.A22, ne.A32'),
                hcat(ne.A13', ne.A32, spzeros(n3, n3))
            )
            ne.b .= vcat(ne.b1, ne.b2, ne.b3)

            return
        end

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

            ManoptExamples.get_right_hand_side_simplified_row!(M, Op, Optrial, eval, btrial_y, ne.integrand_y, id_transport, ne.discrete_time_interval; row_index = 1, test_space = ne.test_spaces.x[1])
            @test size(btrial_y) == (9,)
            ManoptExamples.get_right_hand_side_simplified_row!(M, Op, Optrial, eval, btrial_v, ne.integrand_v, ne.vectortransport, ne.discrete_time_interval; row_index = 2, test_space = ne.test_spaces.x[2])
            @test size(btrial_v) == (6,)
            ManoptExamples.get_right_hand_side_simplified_row!(M, Op, Optrial, eval, btrial_λ, ne.integrand_λ, id_transport, ne.discrete_time_interval; row_index = 3, test_space = ne.test_spaces.x[3])
            @test size(btrial_λ) == (12,)

            @test norm(btrial_y - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) + norm(btrial_y) ≈ norm(btrial_y)
            @test norm(btrial_v - [-3.723280013759479, 0.6081203339171481, 0.2693259668949266, 4.94041726735329, 3.2630082906440867, 2.000346215506468]) + norm(btrial_v) ≈ norm(btrial_v)
            @test norm(btrial_λ - [0.005955875845116053, 0.005384487521366693, -0.0005826837870468032, 0.01725769634426519, 0.005085198152608786, -0.005736900950447689, 0.018388113935373665, -0.0063827151309787805, -0.007164080676918993, 0.007086293436224536, -0.006083425762220901, -0.0020098635135180967]) + norm(btrial_λ) ≈ norm(btrial_λ)

            return vcat(btrial_y, btrial_v, btrial_λ)
        end

        NE = NewtonEq(product, integrand_yλ, integrand_vv, integrand_λv, test_spaces, ansatz_spaces, transport, discrete_time_interval)

        st_res2 = vectorbundle_newton(
            product, TangentBundle(product), NE, y_0; sub_problem = solve_in_basis_repr, sub_state = AllocatingEvaluation(),
            stopping_criterion = (StopAfterIteration(1) | StopWhenChangeLess(product, 1.0e-12; outer_norm = Inf)),
            retraction_method = ProductRetraction(ExponentialRetraction(), ProjectionRetraction(), ExponentialRetraction()),
            stepsize = Manopt.AffineCovariantStepsize(product, θ_des = 1.1, outer_norm = Inf),
            return_state = true
        )
    end
end
