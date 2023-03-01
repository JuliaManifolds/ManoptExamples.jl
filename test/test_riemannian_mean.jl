using Manifolds, Manopt, ManoptExamples, Test

@testset "Riemannian Mean Objective" begin
    M = Euclidean(2)
    data = [[0.0, 1.0], [0.0, -1.0]]
    f = ManoptExamples.RiemannianMeanCost(M, data)
    grad_f = ManoptExamples.RiemannianMeanGradient!!(M, data)
    p = [0.0, 0.0]
    X = zero_vector(M, p)
    @test f(M, p) == 2.0
    @test isapprox(M, p, grad_f(M, p), [0.0, 0.0])
    grad_f(M, X, p)
    @test isapprox(M, p, X, [0.0, 0.0])
    o1 = ManoptExamples.Riemannian_mean_objective(M, data; initial_vector=X)
    @test get_cost(M, o1, p) == f(M, p)
    @test get_gradient(M, o1, p) == grad_f(M, p)
    o2 = ManoptExamples.Riemannian_mean_objective(M, data)
    @test get_cost(M, o2, p) == f(M, p)
    @test get_gradient(M, o2, p) == grad_f(M, p)
    # no manifold given
    o = ManoptExamples.Riemannian_mean_objective(data)
    @test get_cost(M, o, p) == 2.0
    @test isapprox(M, p, get_gradient(M, o, p), zero_vector(M, p))
end
