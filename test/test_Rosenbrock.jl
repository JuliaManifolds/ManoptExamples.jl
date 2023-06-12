using Manopt, ManoptExamples, Manifolds, Test

@testset "Rosenbrock Tests" begin
    M = Euclidean(2)
    a = 100.0
    b = 1.0
    c1 = ManoptExamples.RosenbrockCost(a, b)
    p_star = ManoptExamples.minimizer(c1)
    o1 = ManoptExamples.Rosenbrock_objective(; a=a, b=b)
    @test get_cost(M, o1, p_star) ≈ 0
    @test isapprox(M, p_star, get_gradient(M, o1, p_star), zero_vector(M, p_star))
    o2 = ManoptExamples.Rosenbrock_objective(; a=a, b=b, evaluation=InplaceEvaluation())
    X = zero_vector(M, p_star)
    get_gradient!(M, X, o2, p_star)
    @test isapprox(M, p_star, X, zero_vector(M, p_star))
    @testset "Rosenbrock Metric" begin
        Mrb = MetricManifold(M, ManoptExamples.RosenbrockMetric())
        p = [0.1, 0.1]
        @test local_metric(Mrb, p) == [1.04 -0.2; -0.2 1.0]
        @test inverse_local_metric(Mrb, p) == [1.0 0.2; 0.2 1.04]
        X = [0.2, 0.3]
        Xrb = inverse_local_metric(Mrb, p) * X
        @test change_representer(Mrb, EuclideanMetric(), p, X) == Xrb
        q(t) = p .+ t * [X[1], (X[2] + X[1]^2)]
        @test exp(Mrb, p, X, 0.5) == q(0.5)
        @test exp(Mrb, p, X) == q(1.0)
        p1 = copy(M, p)
        exp!(Mrb, p1, p, X, 0.5)
        @test p1 == q(0.5)
        exp!(Mrb, p1, p, X)
        @test p1 == q(1.0)
        @test isapprox(Mrb, p, log(Mrb, p, p1), X)
        Y = [0.4, 0.5]
        @test inner(Mrb, p, X, Y) == X' * local_metric(Mrb, p) * Y
    end
end
