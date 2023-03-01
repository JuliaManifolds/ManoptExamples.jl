using Manifolds, Manopt, ManoptExamples, Test

@testset "Robust PCA" begin
    M = Grassmann(3, 2)
    p = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    data1 = [1.0 0.0; 0.0 1.0; 0.0 0.0]
    o1 = ManoptExamples.robust_PCA_objective(data1)
    @test get_cost(M, o1, p) == 0
    @test isapprox(M, p, get_gradient(M, o1, p), zero_vector(M, p))
    data2 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    o2 = ManoptExamples.robust_PCA_objective(M, data2; evaluation=InplaceEvaluation())
    @test get_cost(M, o2, p) > 0
    @test isapprox(M, p, get_gradient(M, o2, p), zero_vector(M, p))
    X = zero_vector(M, p)
    get_gradient!(M, X, o2, p)
    @test isapprox(M, p, X, zero_vector(M, p))
end
