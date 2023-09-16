using Manifolds, Manopt, ManoptExamples, Test

@testset "Rayleigh Quotient" begin
    A = [2.0 0.0 0; 0.0 3.0 4.0; 0.0 4.0 9.0]
    λ = 1.0 # Smallest Eigenvalue
    x = [0.0, -2.0, 1.0] # corresponding Eigenvector
    xn = 5.0
    p = x ./ sqrt(xn)
    X = [1.0, 0.0, 0.0]
    f = ManoptExamples.RayleighQuotientCost(A)
    grad_f = ManoptExamples.RayleighQuotientGrad!!(A)
    Hess_f = ManoptExamples.RayleighQuotientHess!!(A)

    M = Sphere(2)
    E = ℝ^3
    @test f(M, p) == 1
    @test f(E, x) == λ * xn
    @test norm(M, p, grad_f(M, p)) ≈ 0.0 atol = 3e-16
    Y = zero_vector(M, p)
    grad_f(M, Y, p)
    @test norm(M, p, Y) ≈ 0.0 atol = 3e-16
    @test grad_f(E, x) == 2 * x
    grad_f(E, Y, x)
    @test Y == 2 * x

    @test Hess_f(M, p, X) == 2 .* X
    Hess_f(M, Y, p, X)
    @test Y == 2 .* X

    @test Hess_f(E, x, X) == 4 .* X
    Hess_f(E, Y, x, X)
    @test Y == 4 .* X
end
