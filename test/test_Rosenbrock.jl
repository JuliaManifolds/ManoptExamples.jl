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
    X = zero_vector(M, p_star)
    get_gradient!(M, X, o1, p_star)
    @test isapprox(M, p_star, X, zero_vector(M, p_star))
end
