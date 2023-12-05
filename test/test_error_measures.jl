using Manifolds, Test, ManoptExamples
using ManoptExamples: mean_squared_error, mean_average_error
using Random
Random.seed!(42)

@testset "Error Measures" begin
    M = Sphere(2)
    N = PowerManifold(M, NestedPowerRepresentation(), 2)
    using Random: seed!
    seed!(42)
    d = Manifolds.uniform_distribution(M, [1.0, 0.0, 0.0])
    w = rand(d)
    x = rand(d)
    y = rand(d)
    z = rand(d)
    a = [w, x]
    b = [y, z]
    @test mean_squared_error(M, x, y) == distance(M, x, y)^2
    @test mean_squared_error(N, a, b) == 1 / 2 * (distance(M, w, y)^2 + distance(M, x, z)^2)
    @test mean_average_error(M, x, y) == distance(M, x, y)
    @test mean_average_error(N, a, b) == 1 / 2 * sum(distance.(Ref(M), a, b))
end
